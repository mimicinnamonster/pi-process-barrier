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
 * - This extension registers a blocking tool named `barrier` EAGERLY
 *   at session start (registration is required for subagent spawn preflight
 *   and for by-name resolution of injected calls) but toggles its ACTIVE
 *   tool-set membership so it is present in the model's callable schema ONLY
 *   exactly while processes are running; on `message_end` for a terminal
 *   assistant message it INJECTS a synthetic toolCall for the tool into that
 *   message (in-place) whenever processes are still running. Because the agent
 *   loop re-reads message.content AFTER extension handlers run and resolves
 *   tool calls by name with no provenance check, the injected call executes
 *   like any other: `hasToolCall` is true at the relay, so the parent never
 *   arms the settle-drain and the child survives.
 * - The barrier tool's `execute()` BLOCKS (keeps the turn's tool batch open)
 *   until every watched process exits (or the timeout passes), then returns a
 *   summary as its tool_result. The loop's automatic follow-up LLM call is
 *   the agent's reaction: with the summary in context it reports, retries, or
 *   starts more processes. When it finally ends its turn with nothing left
 *   running, the run completes normally (the killAll() cleanup at dispose is
 *   harmless because nothing is left running).
 *
 * Guard: the tool is deliberately NOT callable as a general utility. It is
 * registered at session start (so the spawn preflight passes for agents whose
 * `tools:` allowlist includes it and injected calls can resolve) but is absent
 * from the ACTIVE tool set while no processes are running — and therefore
 * excluded from the model's function-calling schema. If an agent nevertheless
 * calls it directly
 * while no processes are running, execute() THROWS so the LLM sees an error
 * result telling it not to call directly. It has no promptSnippet, so it is
 * omitted from the "Available tools" system-prompt section (custom tools only
 * appear there when a snippet is provided).
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

const BARRIER_TOOL_NAME = "barrier";

/**
 * Session-scoped id-set of barrier toolCall ids INJECTED by this
 * extension (never removed: it records provenance for the session's lifetime,
 * so a block decision never depends on the retire/context-scrub lifecycle).
 * The loader re-evaluates this module per session (jiti moduleCache: false),
 * so this never leaks across sessions.
 */
const injectedBarrierCallIds = new Set<string>();

/**
 * Session-scoped id-set of BLOCKED (model-authored) barrier call ids.
 * A blocked call produces an error toolResult with no barrier execution; these
 * ids are hidden from the model-visible context by the context filter below so
 * the model never sees evidence of its attempt. Not evicted for the session.
 */
const blockedBarrierCallIds = new Set<string>();

// Session-scoped one-shot registration guard. The loader re-evaluates this
// module per session (jiti moduleCache: false), so this flag never leaks
// across sessions: the tool is registered at most once per session, eagerly
// at factory init.
let registered = false;

// History-scrub bookkeeping, also session-scoped (same re-evaluation rule as
// `registered`). A barrier wait's toolCall/toolResult pair must stay visible
// to the immediate wake-up follow-up LLM call (that call reads the process
// summary from the toolResult). The pair is SERVED as soon as it is shown to
// that call (a context projection) and RETIRED the moment the wake-up call's
// assistant message is finalized (message_end), so any later LLM call —
// further calls within the same turn or later turns — never sees it. The
// message_start(user) and new-injection retire paths remain as idempotent
// safety nets. Retired pairs are stripped from the model-visible context on
// every subsequent LLM call via the pi context event, so the model never sees
// a past barrier invocation in history and cannot imitate it.
let currentBarrierCallId: string | undefined;
const retiredBarrierCallIds = new Set<string>();

/**
 * The invocation that has already been shown to the wake-up follow-up LLM
 * call via a context projection but whose assistant message has not yet been
 * finalized. From that message's message_end it is retired immediately, so
 * later LLM calls never see it. Always equals currentBarrierCallId while set
 * (set in the context handler, cleared by retireId).
 */
let servedBarrierCallId: string | undefined;

/**
 * Move an id to the hidden set and drop it from the current/served slot(s).
 * Idempotent (Set.add dedupes); safe for undefined; clears both references to
 * the id so a retired invocation can never be re-served or re-injected.
 */
function retireId(id: string | undefined): void {
  if (id === undefined) return;
  retiredBarrierCallIds.add(id);
  if (currentBarrierCallId === id) currentBarrierCallId = undefined;
  if (servedBarrierCallId === id) servedBarrierCallId = undefined;
}

/** Retire the current active invocation (if any): it moves to the hidden set. */
function retireCurrentBarrier(): void {
  retireId(currentBarrierCallId);
}

const DIRECT_CALL_ERROR_MESSAGE =
  "barrier is an internal helper: it only waits for processes started with the process tool. " +
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

/**
 * Add/remove the barrier tool from the session's ACTIVE tool set. Tools outside
 * the active set are excluded from the model's function-calling schema, so this
 * is what keeps the tool invisible to models except exactly while processes are
 * running. Idempotent; never throws (action methods may still be unbound during
 * extension load, and a failed toggle must never break a turn or the relay).
 */
function setBarrierActive(pi: ExtensionAPI, on: boolean): void {
  try {
    const active = pi.getActiveTools();
    const has = active.includes(BARRIER_TOOL_NAME);
    if (on && !has) {
      pi.setActiveTools([...active, BARRIER_TOOL_NAME]);
    } else if (!on && has) {
      pi.setActiveTools(active.filter((n) => n !== BARRIER_TOOL_NAME));
    }
  } catch {
    // Never throw.
  }
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

  // The barrier tool is registered EAGERLY at factory init (see the
  // ensureRegistered call below) — not lazily. It must be in the
  // registered-tool list before any subagent can spawn: pi-subagents validates
  // each agent's frontmatter `tools:` allowlist against REGISTERED tools, and
  // the agents that list barrier (delegate/worker/scout/oracle/
  // worker-full) would fail to spawn with "Agent 'x' requested unavailable
  // child tools: barrier" if it were missing. It must also be
  // registered for auto-injected toolCalls to resolve by name. Visibility is
  // governed separately by the active-set toggle: the tool is only ACTIVE
  // while processes are running. The loader re-evaluates this module per
  // session (jiti moduleCache: false), so the module-level `registered` flag
  // is session-scoped and the tool is registered at most once per session.
  const ensureRegistered = (currentPi: ExtensionAPI): void => {
    if (registered) return;
    try {
      currentPi.registerTool({
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

          try {
            const { entries, checkpoint } = await waitForAllExited(
              pi,
              running,
              deadline,
              signal,
              effCheckIntervalSeconds > 0
                ? effCheckIntervalSeconds * 1000
                : 0,
            );
            if (checkpoint) {
              const elapsedSeconds = Math.max(
                0,
                Math.round((Date.now() - waitStartedAt) / 1000),
              );
              return {
                content: [
                  {
                    type: "text",
                    text: formatCheckpoint(entries, elapsedSeconds),
                  },
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
              content: [
                {
                  type: "text",
                  text: formatReport(entries, timedOutSeconds),
                },
              ],
              details: { processes: entries },
              // Intentionally no `terminate: true`: the automatic follow-up LLM
              // call after this tool batch IS the agent's wake-up reaction.
            };
          } finally {
            // The wait resolved (or was aborted): the barrier is no longer
            // needed in the model's schema. Any still-running processes
            // re-activate it on the next process event or message_end.
            setBarrierActive(pi, false);
          }
        },
      });
      registered = true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(
        `[process-barrier] failed to register ${BARRIER_TOOL_NAME}: ${message}`,
      );
    }
  };

  // Eager registration at factory init, before any handler registration or
  // anything else that could throw: the tool must be in the registered-tool
  // list by the time subagent spawn preflight runs and by the time any
  // auto-injected toolCall resolves. registerTool() is valid during extension
  // load (tools refresh happens on bind). Visibility stays inactive here —
  // only live-process state may add it to the ACTIVE tool set.
  ensureRegistered(pi);

  // Session start: idempotent safety — the barrier must never start ACTIVE
  // (i.e. visible in the model's schema). Only live-process state may activate
  // it. (At the very end of factory init action methods may still be unbound;
  // this runs once the runtime is bound.)
  pi.on("session_start", () => {
    setBarrierActive(pi, false);
  });

  // Session-scope state listener: keep the barrier's visibility in sync with
  // live process state on the shared extension event bus (the same channels
  // used by waitForAllExited). Any live (running|terminating) process makes
  // the tool ACTIVE; once none are live it is hidden again. (Registration
  // already happened at factory init; ensureRegistered here is an idempotent
  // no-op kept for robustness.)
  const syncBarrierState = (payload: unknown): void => {
    try {
      const status = (payload as { status?: unknown } | null)?.status;
      if (typeof status === "string") {
        // processes:ended carries a single ProcessInfo with its final status;
        // processes:started carries one with a live status. Judge by it.
        if (isLive(status)) {
          ensureRegistered(pi);
          setBarrierActive(pi, true);
        } else {
          setBarrierActive(pi, false);
        }
        return;
      }
      // processes:changed carries only { reason: "started"|"ended"|"cleared" } —
      // no process data. Fall back to an authoritative list snapshot.
      void requestList(pi).then((list) => {
        const live = list.some(
          (p) => p && typeof p.id === "string" && isLive(p.status),
        );
        if (live) {
          ensureRegistered(pi);
          setBarrierActive(pi, true);
        } else {
          setBarrierActive(pi, false);
        }
      });
    } catch {
      // Never throw from an event handler.
    }
  };
  const stateListeners: Array<() => void> = [
    pi.events.on("processes:changed", (payload) => {
      syncBarrierState(payload);
    }),
    pi.events.on("processes:ended", (payload) => {
      syncBarrierState(payload);
    }),
  ];
  pi.on("session_shutdown", () => {
    for (const unsub of stateListeners) {
      try {
        unsub();
      } catch {
        // Never throw during shutdown.
      }
    }
  });

  pi.on("message_end", async (event, ctx) => {
    try {
      // A served invocation is retired as soon as the wake-up LLM call's
      // assistant message is finalized — message_end fires for every persisted
      // assistant message (including ones containing further tool calls), so
      // this runs before any LATER LLM call in the same turn (or a later turn)
      // can see the pair. It is a no-op on other message_ends (the barrier's
      // toolResult persistence, the final message that carried the injection):
      // servedBarrierCallId stays undefined until the context handler below
      // has served the pair, so the wake-up call itself still reads the
      // process summary.
      if (servedBarrierCallId !== undefined) {
        retireId(servedBarrierCallId);
      }
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
      if (running.length === 0) {
        // Nothing to wait for: hide the barrier from the model's schema so it
        // cannot be called ad hoc while idle (authoritative re-check after any
        // earlier wait resolved).
        setBarrierActive(pi, false);
        return;
      }

      // A process is still running: make the barrier ACTIVE so the injected
      // toolCall resolves by name and the model can call it. (Registration
      // already happened at factory init; ensureRegistered is an idempotent
      // no-op.)
      ensureRegistered(pi);
      setBarrierActive(pi, true);

      if (!Array.isArray(m.content)) {
        return; // cannot inject a toolCall into non-array content
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
      // A new injected barrier call supersedes any previous one (only one
      // barrier runs at a time): retire the old pair, then mark the new
      // invocation as the CURRENT one (visible to the wake-up follow-up call).
      retireCurrentBarrier();
      currentBarrierCallId = toolCall.id;
      // Record provenance AFTER retiring the previous invocation: the injected
      // id must be recognizable as ours for the tool_call block handler and
      // stay recorded even after the pair is retired from the context scrub.
      injectedBarrierCallIds.add(toolCall.id);
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

  // Block MODEL-AUTHORED barrier calls before they can execute.
  // Tool calls are preflighted (agent-loop prepareToolCall) before the tool's
  // execute() runs; extending the pi `tool_call` event here can block a call
  // with { block: true, reason } (ToolCallEventResult, types.d.ts:778-782).
  // Only the id the extension itself injected (recorded in
  // injectedBarrierCallIds) may run the real wait. Any other id is
  // model-authored (the LLM imitating the tool it briefly saw) and is blocked
  // UNCONDITIONALLY: registration and injection are per-session, and the
  // injected id is ALWAYS recorded (injectedBarrierCallIds.add in the
  // message_end handler) BEFORE the agent loop executes that call — so an
  // unrecorded id is by definition not ours, and there is no legitimate
  // unrecorded case to allow through (no reload edge: the loader re-evaluates
  // this module per session and injection always records first). The tool
  // never executes, no wait starts, no duplicate watch is created. A blocked
  // call leaves an immediate error toolResult; the tool_result tracker below
  // records its id and the context filter hides it, so the model never sees
  // evidence of its own attempt. The execute() guard remains only as a
  // last-resort safety net. Never throws; a block failure must not break the
  // turn.
  pi.on("tool_call", (event) => {
    try {
      const e = event as { toolName?: unknown; toolCallId?: unknown } | null;
      if (!e || e.toolName !== BARRIER_TOOL_NAME) return undefined;
      const id = String(e.toolCallId ?? "");
      if (id === "") return undefined;
      // Provenance: only extension-injected calls may run the wait.
      if (injectedBarrierCallIds.has(id)) return undefined;
      return { block: true, reason: DIRECT_CALL_ERROR_MESSAGE };
    } catch {
      return undefined;
    }
  });

  // A new user turn means any previous barrier wait is done — its summary was
  // already delivered to the wake-up follow-up LLM call. Retire the current
  // invocation: on subsequent LLM calls the context handler below strips the
  // retired pair from the model-visible context.
  pi.on("message_start", (event) => {
    try {
      if (event?.message?.role === "user") retireCurrentBarrier();
    } catch {
      // Never throw from an event handler.
    }
  });

  // Record a BLOCKED (model-authored) barrier call id so the context scrub
  // hides its error toolResult and toolCall block from the model: the goal is
  // that the model never sees evidence it attempted the internal helper.
  pi.on("tool_result", (event) => {
    try {
      const e = event as { toolName?: unknown; toolCallId?: unknown } | null;
      if (!e || e.toolName !== BARRIER_TOOL_NAME) return;
      const id = String(e.toolCallId ?? "");
      if (id === "") return;
      if (injectedBarrierCallIds.has(id)) return;
      // Not ours => a model-authored attempt (blocked pre-execution). Mark its
      // id blocked so the context filter hides its error toolResult.
      blockedBarrierCallIds.add(id);
    } catch {
      // Never throw.
    }
  });

  // Fired before EVERY LLM call (first of a turn AND every follow-up after
  // tool results) with a structuredClone deep copy of the exact messages about
  // to be sent. Strip RETIRED barrier invocations (their toolCall
  // blocks and matching toolResult messages) so the model never sees a past
  // barrier call in history and cannot imitate it. The CURRENT invocation
  // stays visible: the immediate wake-up follow-up call reads the wait summary
  // from its toolResult, and seeing it here counts as SERVING — the next
  // message_end retires it, so no later call (same turn or later) can see or
  // imitate it. The stored session is untouched (pi sessions are append-only —
  // there is no delete/update entry API). Returning undefined is the no-op;
  // never throw (the runtime also guards, but we stay defensive).
  pi.on("context", (event) => {
    try {
      if (
        retiredBarrierCallIds.size === 0 &&
        currentBarrierCallId === undefined
      ) {
        return undefined; // nothing retired, nothing current -> no-op
      }
      const messages = event?.messages;
      if (!Array.isArray(messages)) return undefined;
      const filtered: Array<typeof messages[number]> = [];
      for (const m of messages) {
        const role = (m as { role?: string } | null)?.role;
        if (
          role === "toolResult" &&
          (m as { toolName?: string } | null)?.toolName === BARRIER_TOOL_NAME &&
          (retiredBarrierCallIds.has(
            (m as { toolCallId?: string } | null)?.toolCallId as string,
          ) ||
            // Blocked (model-authored) calls also surface as error toolResults;
            // hide them too so the model never sees evidence of its attempt.
            blockedBarrierCallIds.has(
              (m as { toolCallId?: string } | null)?.toolCallId as string,
            ))
        ) {
          continue; // drop the retired barrier toolResult
        }
        if (role === "assistant" && Array.isArray((m as { content?: unknown }).content)) {
          const original = (
            m as {
              content: Array<{ type?: string; name?: string; id?: string }>;
            }
          ).content;
          const kept = original.filter(
            (c) =>
              !(
                c?.type === "toolCall" &&
                c?.name === BARRIER_TOOL_NAME &&
                (retiredBarrierCallIds.has(String(c?.id)) ||
                  blockedBarrierCallIds.has(String(c?.id)))
              ),
          );
          if (
            servedBarrierCallId === undefined &&
            currentBarrierCallId !== undefined &&
            original.some(
              (c) =>
                c?.type === "toolCall" &&
                c?.name === BARRIER_TOOL_NAME &&
                String(c?.id) === currentBarrierCallId,
            )
          ) {
            // First time the CURRENT (not yet retired) invocation is shown to
            // a context projection: it is being served to the wake-up call and
            // stays visible for THIS call only; the wake-up assistant message's
            // message_end retires it (served-then-retired). The filter above
            // already preserved the block (its id is not retired).
            servedBarrierCallId = currentBarrierCallId;
          }
          if (
            kept.length === 0 &&
            original.every((c) => c?.type === "toolCall")
          ) {
            // Only retired toolCall blocks (no text left): skip the whole
            // message instead of emitting empty content, which some providers
            // reject.
            continue;
          }
          filtered.push({ ...(m as object), content: kept } as never);
          continue;
        }
        filtered.push(m);
      }
      return { messages: filtered as typeof messages };
    } catch {
      // Never throw from a context handler: a failure must not break the LLM
      // call. Returning undefined leaves the context untouched (no-op).
      return undefined;
    }
  });

  // Very end of factory init: the tool must not start ACTIVE (visible in the
  // model's schema). When factory init still runs, action methods are unbound
  // stubs and the helper swallows their throw; session_start repeats this once
  // the runtime is bound.
  setBarrierActive(pi, false);
}