/**
 * process-barrier
 *
 * Keeps a pi-subagents child run alive while spawned processes (managed by the
 * @aliou/pi-processes extension) are still running, and gives the agent a
 * reaction turn when they finish.
 *
 * Mechanism (verified against installed sources):
 * - pi-subagents children are in-process AgentSessions (mode "print"). The
 *   parent computes `terminalAssistantStop` ONLY from the RELAYED child
 *   `message_end` event and then arms a settle-drain (1s grace -> abort ->
 *   3s hard finish -> dispose -> session_shutdown -> @aliou cleanup killAll()).
 *   The drain arms only when the final assistant message has NO tool calls
 *   (`stopReason === "stop" && !hasToolCall`).
 * - This extension registers a blocking tool named `process_barrier` and, on
 *   `message_end` for a terminal assistant message, INJECTS a synthetic
 *   toolCall for it into that message (in-place) whenever processes are still
 *   running. Because the agent loop re-reads message.content AFTER extension
 *   handlers run and resolves tool calls by name with no provenance check,
 *   the injected call executes like any other: `hasToolCall` is true at the
 *   relay, so the parent never arms the settle-drain and the child survives.
 * - The barrier tool's `execute()` BLOCKS (keeps the turn's tool batch open)
 *   until every watched process exits (or the timeout passes), then returns a
 *   summary as its tool_result. The loop's automatic follow-up LLM call is
 *   the agent's reaction: with the summary in context it reports, retries, or
 *   starts more processes. When it finally ends its turn with nothing left
 *   running, the run completes normally (the killAll() cleanup at dispose is
 *   harmless because nothing is left running).
 *
 * Guard: the tool is deliberately NOT callable as a general utility. If an
 * agent calls it directly while no processes are running, execute() THROWS so
 * the LLM sees an error result telling it not to call directly. It has no
 * promptSnippet, so it is omitted from the "Available tools" system-prompt
 * section (custom tools only appear there when a snippet is provided).
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

import { type Static, Type } from "typebox";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_TIMEOUT_SECONDS = 600;
const MAX_TIMEOUT_SECONDS = 1800;
const LIST_FALLBACK_MS = 1000; // no reply -> assume no manager (extension not loaded)
const POLL_INTERVAL_MS = 2500; // safety-net poll while blocked
const DEFAULT_CHECK_INTERVAL_MS = 300_000; // 5 min: wake the agent to verify nothing is hanging
const MIN_CHECK_INTERVAL_MS = 5_000; // sanity floor: keep interplay with the follow-up call sane
const MAX_CHECK_INTERVAL_SECONDS = 86_400; // 24h cap for the settings value

/**
 * Resolve pi's agent settings directory. Mirrors pi core's getAgentDir()
 * (dist/config.ts: PI_CODING_AGENT_DIR override, else ~/.pi/agent).
 */
function getAgentDir(): string {
  const envDir = process.env.PI_CODING_AGENT_DIR;
  if (envDir) {
    return envDir === "~" ? homedir() : envDir.replace(/^~\//, `${homedir()}/`);
  }
  return join(homedir(), ".pi", "agent");
}

/**
 * Read extension config from pi settings.json. pi core exposes no settings
 * API to extensions; the established pattern (pi-subagents reads the
 * "subagents" key, @aliou/pi-processes uses a config loader over the same
 * file) is to read the file directly at factory init.
 *
 * Expected shape (top-level key "processBarrier"):
 *   { "processBarrier": { "checkIntervalSeconds": 300 } }
 * Absent key/invalid value -> fallback (default).
 */
function readSettings(): { checkIntervalSeconds?: number } {
  try {
    const raw = readFileSync(join(getAgentDir(), "settings.json"), "utf-8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const cfg = parsed?.processBarrier;
    if (!cfg || typeof cfg !== "object" || Array.isArray(cfg)) return {};
    const rawInterval = (cfg as Record<string, unknown>).checkIntervalSeconds;
    if (typeof rawInterval !== "number" || !Number.isFinite(rawInterval)) {
      return {};
    }
    return { checkIntervalSeconds: rawInterval };
  } catch {
    return {};
  }
}

/** Parse an integer env var, falling back when unset/invalid. */
function readEnvInt(name: string, fallback: number): number {
  try {
    const raw = process.env[name];
    if (raw === undefined || raw === "") return fallback;
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : fallback;
  } catch {
    return fallback;
  }
}

const BARRIER_TOOL_NAME = "process_barrier";

const DIRECT_CALL_ERROR_MESSAGE =
  "process_barrier is an internal helper: it only waits for processes started with the process tool. " +
  "There are currently no running processes, so there is nothing to wait for. Do not call it directly.";

/** Minimal structural view of the @aliou/pi-processes ProcessInfo payload. */
interface ProcessInfo {
  id: string;
  name: string;
  pid: number;
  status: string;
  exitCode: number | null;
  success: boolean | null;
  endReason: string | null;
  signal: { name: string; number: number | null; description: string } | null;
  errorMessage: string | null;
  /** Epoch ms when the process started (when the manager provides it). */
  startTime: number | null;
}

interface ReportEntry {
  id: string;
  name: string;
  status: string; // exited | killed | terminate_timeout | running | terminating | still_running
  exitCode: number | null;
  success: boolean | null;
  endReason: string | null;
  signalName: string | null;
  errorMessage: string | null;
  startTime: number | null;
}

/** Outcome of waiting: either the final report or an early check-in wake-up. */
interface WaitResult {
  entries: ReportEntry[];
  checkpoint: boolean;
}

/** Statuses that count as still alive at snapshot time. */
function isLive(status: string): boolean {
  return status === "running" || status === "terminating";
}

/** Statuses that count as finished (no longer live). */
function isDone(status: string): boolean {
  return (
    status === "exited" || status === "killed" || status === "terminate_timeout"
  );
}

/**
 * Ask the process manager for the current process list by emitting a sync
 * request over the shared extension event bus (same channel/handler shape the
 * @aliou extension itself uses). If the manager is not loaded in this session,
 * the reply never arrives and we fall back to [].
 */
function requestList(pi: ExtensionAPI): Promise<ProcessInfo[]> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (list: unknown): void => {
      if (settled) return;
      settled = true;
      resolve(Array.isArray(list) ? (list as ProcessInfo[]) : []);
    };
    try {
      pi.events.emit("processes:request:list", {
        reply: (list: ProcessInfo[]) => finish(list),
      });
    } catch {
      finish([]);
    }
    setTimeout(() => finish([]), LIST_FALLBACK_MS);
  });
}

function toReportEntry(p: ProcessInfo): ReportEntry {
  return {
    id: p.id,
    name: p.name,
    status: p.status,
    exitCode: p.exitCode,
    success: p.success,
    endReason: p.endReason,
    signalName: p.signal?.name ?? null,
    errorMessage: p.errorMessage,
    startTime: p.startTime ?? null,
  };
}

function formatReport(entries: ReportEntry[], timedOutSeconds: number): string {
  const stillRunning = entries.filter((e) => e.status === "still_running");
  const header =
    stillRunning.length > 0
      ? `Timed out after ${timedOutSeconds}s; processes still running:`
      : "All processes finished:";
  const lines = entries.map((e) => {
    let detail: string;
    switch (e.status) {
      case "exited":
        detail = `exited (code ${e.exitCode ?? "?"}, ${e.success ? "success" : "failed"})`;
        break;
      case "killed":
        detail = e.signalName
          ? `killed (signal ${e.signalName})`
          : "killed";
        if (e.endReason) detail += ` (reason: ${e.endReason})`;
        break;
      case "terminate_timeout":
        detail = "terminate_timeout (kill did not complete in time)";
        break;
      default:
        detail = "still running (timeout)";
    }
    if (e.errorMessage) detail += ` (error: ${e.errorMessage})`;
    return `- ${e.id} "${e.name}" ${detail}`;
  });
  return [header, ...lines].join("\n");
}

/**
 * Check-in formatter: the barrier resolved early (interval elapsed, processes
 * still running) to wake the agent for a periodic health check. Shows live
 * per-process status and uptime when the manager provides startTime.
 */
function formatCheckpoint(entries: ReportEntry[], elapsedSeconds: number): string {
  const lines = entries.map((e) => {
    const uptime =
      typeof e.startTime === "number" && e.startTime > 0
        ? `, uptime: ${Math.max(0, Math.round((Date.now() - e.startTime) / 1000))}s`
        : "";
    return `- ${e.id} \"${e.name}\" (${e.status}${uptime})`;
  });
  return [
    `Still waiting after ${elapsedSeconds}s; processes still running:`,
    ...lines,
    "Check whether any process is hanging (process list / process output), then simply end your turn: the barrier will keep waiting.",
  ].join("\n");
}

/**
 * Block until every watched process reaches a final status (exited, killed,
 * terminate_timeout), the deadline passes, or the execute() abort signal
 * fires. Uses the manager's live `processes:ended` broadcasts plus a polling
 * safety net (also re-checks on `processes:changed`).
 */
function waitForAllExited(
  pi: ExtensionAPI,
  watched: ProcessInfo[],
  deadline: number,
  signal: AbortSignal | undefined,
  checkIntervalMs = 0,
): Promise<WaitResult> {
  return new Promise((resolve) => {
    const watchIds = new Set(watched.map((p) => p.id));
    const report = new Map<string, ReportEntry>();
    const startTime = Date.now();
    let latestList: ProcessInfo[] = [];
    let settled = false;
    let scheduled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubs: Array<() => void> = [];

    const finish = (result: WaitResult): void => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(result);
    };

    const cleanup = (): void => {
      if (timer) clearTimeout(timer);
      for (const unsub of unsubs) unsub();
      signal?.removeEventListener("abort", onAbort);
    };

    const record = (info: ProcessInfo): void => {
      if (
        info &&
        typeof info.id === "string" &&
        watchIds.has(info.id) &&
        !report.has(info.id) &&
        isDone(info.status)
      ) {
        report.set(info.id, toReportEntry(info));
      }
    };

    const onAbort = (): void => {
      finish({ entries: finalize(), checkpoint: false });
    };

    const finalize = (): ReportEntry[] => {
      const entries: ReportEntry[] = [];
      for (const p of watched) {
        entries.push(
          report.get(p.id) ?? { ...toReportEntry(p), status: "still_running" },
        );
      }
      return entries;
    };

    const allDone = (): boolean => {
      for (const p of watched) {
        if (!report.has(p.id)) return false;
      }
      return true;
    };

    const checkpointEntries = (): ReportEntry[] => {
      const byId = new Map(latestList.map((p) => [p.id, p] as const));
      return watched.map((p) => {
        const live = byId.get(p.id);
        const base = live ?? p;
        return {
          ...toReportEntry(base),
          status: isLive(base.status) ? base.status : "still_running",
        };
      });
    };

    const tick = async (): Promise<void> => {
      if (settled) return;
      const list = await requestList(pi);
      if (settled) return;
      latestList = list;
      for (const p of list) record(p);
      if (allDone()) {
        finish({ entries: finalize(), checkpoint: false });
        return;
      }
      if (Date.now() >= deadline) {
        finish({ entries: finalize(), checkpoint: false });
        return;
      }
      if (checkIntervalMs > 0 && Date.now() - startTime >= checkIntervalMs) {
        // Early resolve: the agent is woken for a health check; its next
        // terminal turn re-injects the barrier and the wait continues.
        finish({ entries: checkpointEntries(), checkpoint: true });
        return;
      }
      scheduled = false;
      schedule();
    };

    const schedule = (): void => {
      if (settled || scheduled || timer) return;
      scheduled = true;
      timer = setTimeout(() => {
        timer = undefined;
        void tick();
      }, POLL_INTERVAL_MS);
    };

    unsubs.push(
      pi.events.on("processes:ended", (data) => {
        record(data as ProcessInfo);
        if (allDone()) finish({ entries: finalize(), checkpoint: false });
      }),
      pi.events.on("processes:changed", () => {
        // A process may have ended without us catching the individual event;
        // trigger an immediate safety-net check.
        if (!settled && !timer) void tick();
      }),
    );

    if (signal?.aborted) {
      finish({ entries: finalize(), checkpoint: false });
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });

    // One immediate check: snapshot could already be stale (process exited
    // between the initial snapshot and now).
    void tick();
  });
}

const BarrierParams = Type.Object({
  timeoutSeconds: Type.Optional(
    Type.Number({
      description: "Max seconds to wait (default 600, max 1800).",
    }),
  ),
  checkIntervalSeconds: Type.Optional(
    Type.Number({
      description:
        "Direct-call override: seconds to wait before waking the agent for a periodic health check (0 disables). " +
        "On automatic injection, the pi setting processBarrier.checkIntervalSeconds is used " +
        "(fallback: env PI_PROCESS_BARRIER_CHECK_INTERVAL_MS, default 300).",
    }),
  ),
});

export default function (pi: ExtensionAPI): void {
  // Factory-level config, read once at extension init. Precedence (highest wins):
  // 1. pi settings key "processBarrier" -> { checkIntervalSeconds } (0 = disabled)
  // 2. PI_PROCESS_BARRIER_CHECK_INTERVAL_MS env var (0 = disabled)
  // 3. default 300s. A direct `checkIntervalSeconds` tool param still
  //    overrides all of these on manual calls.
  const settingsInterval = readSettings().checkIntervalSeconds;
  let checkIntervalSeconds: number;
  if (
    settingsInterval !== undefined &&
    Number.isFinite(settingsInterval) &&
    settingsInterval !== 0
  ) {
    // Clamp a positive settings value to a sane range; 0 must stay disabled.
    checkIntervalSeconds =
      Math.min(Math.max(settingsInterval, MIN_CHECK_INTERVAL_MS / 1000), MAX_CHECK_INTERVAL_SECONDS);
  } else {
    const rawCheckIntervalMs = readEnvInt(
      "PI_PROCESS_BARRIER_CHECK_INTERVAL_MS",
      DEFAULT_CHECK_INTERVAL_MS,
    );
    const checkIntervalMs =
      rawCheckIntervalMs > 0
        ? Math.max(rawCheckIntervalMs, MIN_CHECK_INTERVAL_MS)
        : 0;
    checkIntervalSeconds = checkIntervalMs > 0 ? checkIntervalMs / 1000 : 0;
  }
  // PI_PROCESS_BARRIER_TIMEOUT_SECONDS: overrides the default max wait (600s);
  // always capped at MAX_TIMEOUT_SECONDS (1800s).
  const defaultTimeoutSeconds = Math.min(
    Math.max(
      readEnvInt("PI_PROCESS_BARRIER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
      1,
    ),
    MAX_TIMEOUT_SECONDS,
  );

  pi.registerTool({
    name: BARRIER_TOOL_NAME,
    label: "Wait for Processes",
    description:
      "Blocks until all processes started with the process tool have exited, then reports their final statuses. " +
      "It is automatically injected when your turn ends while processes are still running, so you will be woken with the results. " +
      "If a configurable check-in interval has elapsed while processes still run, you will be woken periodically to verify processes aren't hanging; " +
      "it is safe to keep your turn short (check process list/output, decide, end your turn) and the barrier will continue waiting. " +
      "The interval is configurable in pi settings (processBarrier.checkIntervalSeconds; 0 disables periodic check-ins). " +
      "Do not call it directly unless you deliberately want to block until processes finish; if no processes are running it will error.",
    // No promptSnippet: custom tools without one are omitted from the
    // "Available tools" section of the default system prompt (verified in
    // dist/core/system-prompt.js). The tool is auto-injected, not meant for
    // ad-hoc use.
    parameters: BarrierParams,
    async execute(
      _toolCallId: string,
      params: Static<typeof BarrierParams>,
      signal?: AbortSignal,
    ) {
      const all = await requestList(pi);
      const running = all.filter(
        (p) => p && typeof p.id === "string" && isLive(p.status),
      );
      if (running.length === 0) {
        // The agent (or anything else) called this tool directly with nothing
        // to wait for. Throw -> the agent loop encodes it as an error tool
        // result so the LLM sees the message above instead of a fake success.
        throw new Error(DIRECT_CALL_ERROR_MESSAGE);
      }

      const timeoutSeconds = Math.min(
        Math.max(params.timeoutSeconds ?? defaultTimeoutSeconds, 1),
        MAX_TIMEOUT_SECONDS,
      );
      const effCheckIntervalSeconds = Math.max(
        params.checkIntervalSeconds ?? checkIntervalSeconds,
        0,
      );
      const waitStartedAt = Date.now();
      const deadline = Date.now() + timeoutSeconds * 1000;
      console.error(
        `[process-barrier] waiting for ${running.length} process(es): ${running.map((p) => p.id).join(", ")} (timeout ${timeoutSeconds}s, check-in ${effCheckIntervalSeconds}s)`,
      );

      const { entries, checkpoint } = await waitForAllExited(
        pi,
        running,
        deadline,
        signal,
        effCheckIntervalSeconds > 0 ? effCheckIntervalSeconds * 1000 : 0,
      );
      if (checkpoint) {
        const elapsedSeconds = Math.max(
          0,
          Math.round((Date.now() - waitStartedAt) / 1000),
        );
        return {
          content: [
            { type: "text", text: formatCheckpoint(entries, elapsedSeconds) },
          ],
          details: { checkpoint: true, processes: entries },
          // Intentionally no `terminate: true`: the follow-up LLM call after
          // this tool batch is the periodic check-in reaction; when the agent
          // ends its turn again the barrier is re-injected for a fresh wait.
        };
      }
      const timedOutSeconds = Math.max(
        0,
        Math.round((deadline - Date.now()) / 1000),
      );
      return {
        content: [{ type: "text", text: formatReport(entries, timedOutSeconds) }],
        details: { processes: entries },
        // Intentionally no `terminate: true`: the automatic follow-up LLM
        // call after this tool batch IS the agent's wake-up reaction.
      };
    },
  });

  pi.on("message_end", async (event, ctx) => {
    try {
      const m = event?.message as {
        role?: string;
        stopReason?: string;
        content?: unknown[] | string | null;
      } | null;
      // Only terminal assistant messages: last model turn, no pending tool calls.
      if (!m || m.role !== "assistant" || m.stopReason !== "stop") return;
      // Defensive: never inject into truncated/failed/aborted turns.
      if (
        m.stopReason === "length" ||
        m.stopReason === "error" ||
        m.stopReason === "aborted"
      ) {
        return;
      }
      if (
        Array.isArray(m.content) &&
        m.content.some((c) => c && (c as { type?: string }).type === "toolCall")
      ) {
        return;
      }

      // Snapshot currently live processes.
      const all = await requestList(pi);
      const running = all.filter(
        (p) => p && typeof p.id === "string" && isLive(p.status),
      );
      if (running.length === 0 || !Array.isArray(m.content)) {
        return; // nothing to wait for — no overhead
      }

      // Inject a synthetic toolCall for the barrier tool. The agent loop
      // re-reads message.content after extension handlers run and resolves
      // tool calls by name (no provenance check), so this executes like a
      // normal call; its blocking execute() keeps the run alive until every
      // watched process exits.
      const toolCall = {
        type: "toolCall",
        id: `${BARRIER_TOOL_NAME}_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`,
        name: BARRIER_TOOL_NAME,
        arguments: {},
      };
      m.content.push(toolCall as never);
      console.error(
        `[process-barrier] injected ${BARRIER_TOOL_NAME} for ${running.length} running process(es): ${running.map((p) => p.id).join(", ")} (check-in ${checkIntervalSeconds}s)`,
      );
    } catch (err) {
      // Never throw from a message_end handler: a failure here must not break
      // the turn or the relay. Surface what we know and return.
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[process-barrier] error: ${message}`);
      const m = event?.message as { content?: unknown[] | string | null } | null;
      if (m) {
        const note = {
          type: "text",
          text: `[process-barrier] failed while waiting for processes: ${message}`,
        };
        if (Array.isArray(m.content)) m.content.push(note as never);
        else m.content = [note as never];
      }
    }
  });
}