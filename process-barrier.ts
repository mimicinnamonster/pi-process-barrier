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

import { type Static, Type } from "typebox";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_TIMEOUT_SECONDS = 600;
const MAX_TIMEOUT_SECONDS = 1800;
const LIST_FALLBACK_MS = 1000; // no reply -> assume no manager (extension not loaded)
const POLL_INTERVAL_MS = 2500; // safety-net poll while blocked
const MAX_POLLS = 100;

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
}

interface ReportEntry {
  id: string;
  name: string;
  status: string; // exited | killed | terminate_timeout | still_running
  exitCode: number | null;
  success: boolean | null;
  endReason: string | null;
  signalName: string | null;
  errorMessage: string | null;
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
): Promise<ReportEntry[]> {
  return new Promise((resolve) => {
    const watchIds = new Set(watched.map((p) => p.id));
    const report = new Map<string, ReportEntry>();
    let settled = false;
    let scheduled = false;
    let pollCount = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubs: Array<() => void> = [];

    const finish = (entries: ReportEntry[]): void => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(entries);
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
      finish(finalize());
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

    const tick = async (): Promise<void> => {
      if (settled) return;
      pollCount += 1;
      if (pollCount > MAX_POLLS) {
        finish(finalize());
        return;
      }
      const list = await requestList(pi);
      if (settled) return;
      for (const p of list) record(p);
      if (allDone()) {
        finish(finalize());
        return;
      }
      if (Date.now() >= deadline) {
        finish(finalize());
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
        if (allDone()) finish(finalize());
      }),
      pi.events.on("processes:changed", () => {
        // A process may have ended without us catching the individual event;
        // trigger an immediate safety-net check.
        if (!settled && !timer) void tick();
      }),
    );

    if (signal?.aborted) {
      finish(finalize());
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
});

export default function (pi: ExtensionAPI): void {
  pi.registerTool({
    name: BARRIER_TOOL_NAME,
    label: "Wait for Processes",
    description:
      "Blocks until all processes started with the process tool have exited, then reports their final statuses. " +
      "It is automatically injected when your turn ends while processes are still running, so you will be woken with the results. " +
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
        Math.max(params.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS, 1),
        MAX_TIMEOUT_SECONDS,
      );
      const deadline = Date.now() + timeoutSeconds * 1000;
      console.error(
        `[process-barrier] waiting for ${running.length} process(es): ${running.map((p) => p.id).join(", ")} (timeout ${timeoutSeconds}s)`,
      );

      const entries = await waitForAllExited(pi, running, deadline, signal);
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
        `[process-barrier] injected ${BARRIER_TOOL_NAME} for ${running.length} running process(es): ${running.map((p) => p.id).join(", ")}`,
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