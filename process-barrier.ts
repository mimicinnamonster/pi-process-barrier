/**
 * process-barrier
 *
 * Keeps a pi-subagents child run alive while spawned processes (managed by the
 * @aliou/pi-processes extension) are still running, and gives the agent a
 * reaction turn when they finish.
 *
 * Mechanism (verified against installed pi sources and proven end-to-end by
 * the poc-block-wait experiment):
 * - pi-subagents children are in-process AgentSessions. The parent decides an
 *   early settle from the child's RELAYED `message_end` event: when the final
 *   assistant message has no tool calls (`stopReason === "stop"`), a settle
 *   drain is armed (grace -> abort -> hard finish -> session_shutdown ->
 *   @aliou cleanup killAll()).
 * - pi core AWAITS extension event handlers before forwarding the event to
 *   public listeners (dist/core/agent-session.js), and pi-subagents is such a
 *   public listener. A BLOCKING `message_end` handler therefore defers the
 *   settle decision for as long as the handler runs.
 * - On a terminal assistant message, this extension checks the process
 *   manager for live processes. If any are running, the handler BLOCKS until
 *   every one of them exits (event-driven wake via `processes:ended`, with a
 *   poll safety net), then queues a report as a follow-up user message
 *   (sendUserMessage(..., { deliverAs: "followUp" })). That message forces a
 *   reaction LLM turn in which the agent reports the results; when it ends
 *   its turn with nothing left running, the handler is a fast no-op and the
 *   run settles normally.
 *
 * Why no registered tool: the previous design registered a blocking tool and
 * injected a synthetic toolCall for it into the final assistant message. That
 * required the tool to exist in every child's `tools:` allowlist (spawn
 * preflight + by-name resolution), which leaked the tool's name and schema
 * into the model's world — weak models then called it directly. The
 * allowlist cannot be bypassed (strict registry filter; injected calls
 * resolve only against the active toolset), so the tool approach is
 * unfixable at the extension level. This design registers NOTHING: there is
 * no tool name to leak, no allowlist coupling, and nothing for a model to
 * call or imitate. Leak-free by construction.
 *
 * Periodic check-ins: while processes keep running, the wait can resolve
 * early every `checkIntervalSeconds` (default 120, 0 disables) to wake the
 * agent for a health check. This is also the liveness signal on long holds:
 * each check-in is real LLM activity, which resets pi-subagents' activity
 * tracking. The next terminal message re-enters the wait.
 *
 * Abort: the handler honors the session abort signal (ctx.getSignal()); on
 * abort it stops waiting immediately and does NOT queue a follow-up (the run
 * is terminating; the parent owns the outcome). The wiring to the runner's
 * deadline kill is verified end-to-end in the installed sources: run timeout
 * -> terminateForTimeout (pi-subagents run-child-session.js:272) ->
 * abortChild (:119-133) -> session.abort() (pi agent-session.js:1608-1618)
 * -> agent.abort() -> activeRun.abortController.abort() (pi-agent-core
 * agent.js:212-218) — the same controller ctx.getSignal() returns
 * (agent-session.js:2449). CAVEAT (2026-09-24 incident, run 063bb4e7): the
 * delivery is NOT durable in practice — abortChild is fire-and-forget
 * (`void session.abort().catch(...)`), early-returns when
 * settled/promptSettled, and a 3s abortSettleTimer force-settles the run
 * regardless (ABORT_SETTLE_MS, run-child-session.js:35). In that incident a
 * healthy held turn was killed by the run deadline without the abort ever
 * surfacing here (no "hold end cause=abort" line), and the completed
 * message was lost from every mirror because pi persists the session only
 * AFTER extension handlers return (agent-session.js:580->581->595). That is
 * an upstream pi-subagents reliability gap: a deadline kill should deliver
 * an awaited abort to the child session and let in-flight extension
 * message_end handlers finish (so the completed turn is persisted) before
 * settling. Locally we only shrink the exposure — shorter check-ins
 * (default 120s) and the "gate msg" proof line below.
 *
 * Config (pi settings.json, top-level key "processBarrier", or env fallback):
 *   { "processBarrier": { "checkIntervalSeconds": 120, "waitTimeoutSeconds": 600 } }
 *   PI_PROCESS_BARRIER_CHECK_INTERVAL_MS  (ms, 0 disables check-ins)
 *   PI_PROCESS_BARRIER_TIMEOUT_SECONDS    (hard cap per wait, default 600, max 1800)
 *
 * Logging (observability only, never affects behavior): console.error is not
 * captured anywhere durable (runner.stderr.log stays empty in every run dir),
 * so after a run finishes there is no way to prove whether the barrier held,
 * skipped, or failed to deliver. Every gate/hold/delivery decision is
 * therefore appended (one timestamped line per event) to a durable log file:
 * alongside the child's session.jsonl (async run sessions are named by run
 * id, so the log stays correlate-able), falling back to
 * os.tmpdir()/pi-process-barrier/session-<pid>.log for sessions without a
 * session file. The log is size-capped (~256KB, oldest lines truncated) and
 * every write is failure-safe: logging problems are swallowed and can never
 * alter wait/wake/delivery behavior.
 */

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { homedir, tmpdir } from "node:os";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const DEFAULT_WAIT_TIMEOUT_SECONDS = 600;
const MAX_WAIT_TIMEOUT_SECONDS = 1800;
const LIST_FALLBACK_MS = 1000; // no reply -> assume no manager (extension not loaded)
const LIST_RETRY_TIMEOUT_MS = 500; // re-check window per turn-end gate retry
const LIST_RETRY_MAX_ATTEMPTS = 3; // total attempts at the turn-end gate (incl. the first)
const POLL_INTERVAL_MS = 2500; // safety-net poll while blocked
const DEFAULT_CHECK_INTERVAL_MS = 120_000; // 2 min: wake the agent to verify nothing is hanging; shrinks the window a hold can outlive a run deadline in
const MIN_CHECK_INTERVAL_MS = 5_000; // sanity floor: keep interplay with the follow-up call sane
const MAX_CHECK_INTERVAL_SECONDS = 86_400; // 24h cap for the settings value

// Durable log caps: truncate oldest lines past the byte cap, keeping roughly
// the newest half (aligned to whole lines).
const LOG_FILE_NAME = "process-barrier.log";
const LOG_MAX_BYTES = 256 * 1024;
const LOG_TRIM_KEEP_BYTES = 128 * 1024;

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
 *   { "processBarrier": { "checkIntervalSeconds": 120, "waitTimeoutSeconds": 600 } }
 * Absent key/invalid value -> fallback (default).
 */
function readSettings(): {
  checkIntervalSeconds?: number;
  waitTimeoutSeconds?: number;
} {
  try {
    const raw = readFileSync(join(getAgentDir(), "settings.json"), "utf-8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const cfg = parsed?.processBarrier;
    if (!cfg || typeof cfg !== "object" || Array.isArray(cfg)) return {};
    const out: { checkIntervalSeconds?: number; waitTimeoutSeconds?: number } =
      {};
    const rawInterval = (cfg as Record<string, unknown>).checkIntervalSeconds;
    if (typeof rawInterval === "number" && Number.isFinite(rawInterval)) {
      out.checkIntervalSeconds = rawInterval;
    }
    const rawTimeout = (cfg as Record<string, unknown>).waitTimeoutSeconds;
    if (typeof rawTimeout === "number" && Number.isFinite(rawTimeout)) {
      out.waitTimeoutSeconds = rawTimeout;
    }
    return out;
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

/**
 * Session-scoped flag: set once the process manager has answered ANY list
 * request this session. Lets the turn-end gate retry (see requestListAtTurnEnd)
 * distinguish a manager that is present but slow (retry is worth it) from one
 * that was never loaded (keep the old single-shot latency). The loader
 * re-evaluates this module per session (jiti moduleCache: false), so this
 * never leaks across sessions.
 */
let managerReplied = false;

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
  /** Absolute path to the process's stdout log (when the manager provides it). */
  stdoutFile?: string;
  /** Absolute path to the process's stderr log (when the manager provides it). */
  stderrFile?: string;
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

/** What ended a wait (observability only; logged at hold end). */
type WaitCause = "processes:ended" | "poll" | "cap" | "abort" | "checkpoint";

/** Outcome of waiting: either the final report or an early check-in wake-up. */
interface WaitResult {
  entries: ReportEntry[];
  checkpoint: boolean;
  cause: WaitCause;
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

/** Outcome of one list request: the list plus whether the manager answered. */
interface ListAttempt {
  list: ProcessInfo[];
  /** False when the request timed out (or the bus threw) instead of replying. */
  replied: boolean;
}

/**
 * Emit one bounded list request. `timeoutMs` caps how long we wait for the
 * manager's reply; on timeout/error the attempt resolves with replied=false
 * (empty list) instead of throwing.
 */
function requestListAttempt(
  pi: ExtensionAPI,
  timeoutMs: number,
): Promise<ListAttempt> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (list: unknown, replied: boolean): void => {
      if (settled) return;
      settled = true;
      if (replied) managerReplied = true;
      resolve({
        list: Array.isArray(list) ? (list as ProcessInfo[]) : [],
        replied,
      });
    };
    try {
      pi.events.emit("processes:request:list", {
        reply: (list: ProcessInfo[]) => finish(list, true),
      });
    } catch {
      finish([], false);
    }
    setTimeout(() => finish([], false), timeoutMs);
  });
}

/**
 * Ask the process manager for the current process list by emitting a sync
 * request over the shared extension event bus (same channel/handler shape the
 * @aliou extension itself uses). If the manager is not loaded in this session,
 * the reply never arrives and we fall back to [].
 */
function requestList(pi: ExtensionAPI): Promise<ProcessInfo[]> {
  return requestListAttempt(pi, LIST_FALLBACK_MS).then((a) => a.list);
}

/**
 * Turn-end gate variant used by the message_end handler. A single timed-out
 * request must not be read as "nothing running": a slow manager reply (e.g.
 * under load) would skip the wait, the parent's settle-drain would arm on a
 * live run, and the run would be recorded as failed — the exact wake-abort
 * failure this extension exists to prevent. Retry a bounded number of times,
 * but ONLY while the manager has answered at least once this session, so
 * sessions without the manager keep their old single-shot latency.
 */
async function requestListAtTurnEnd(pi: ExtensionAPI): Promise<ListAttempt> {
  let attempt = await requestListAttempt(pi, LIST_FALLBACK_MS);
  let tries = 1;
  while (!attempt.replied && managerReplied && tries < LIST_RETRY_MAX_ATTEMPTS) {
    attempt = await requestListAttempt(pi, LIST_RETRY_TIMEOUT_MS);
    tries++;
  }
  return attempt;
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
 * Check-in formatter: the wait resolved early (interval elapsed, processes
 * still running) to wake the agent for a periodic health check. Shows live
 * per-process status and uptime when the manager provides startTime.
 */
function formatCheckpoint(
  entries: ReportEntry[],
  elapsedSeconds: number,
): string {
  const lines = entries.map((e) => {
    const uptime =
      typeof e.startTime === "number" && e.startTime > 0
        ? `, uptime: ${Math.max(0, Math.round((Date.now() - e.startTime) / 1000))}s`
        : "";
    return `- ${e.id} "${e.name}" (${e.status}${uptime})`;
  });
  return [
    `Still waiting after ${elapsedSeconds}s; processes still running:`,
    ...lines,
    "Check whether any process is hanging (process list / process output), then simply end your turn: the wait continues automatically.",
  ].join("\n");
}

/**
 * Block until every watched process reaches a final status (exited, killed,
 * terminate_timeout), the deadline passes, or the abort signal fires. Uses
 * the manager's live `processes:ended` broadcasts plus a polling safety net
 * (also re-checks on `processes:changed`). Resolves early with
 * checkpoint=true when checkIntervalMs elapses while processes still run.
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

    const finish = (result: WaitResult, cause: WaitCause): void => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve({ ...result, cause });
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
      finish({ entries: finalize(), checkpoint: false }, "abort");
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
        finish({ entries: finalize(), checkpoint: false }, "poll");
        return;
      }
      if (Date.now() >= deadline) {
        finish({ entries: finalize(), checkpoint: false }, "cap");
        return;
      }
      if (checkIntervalMs > 0 && Date.now() - startTime >= checkIntervalMs) {
        // Early resolve: the agent is woken for a health check; its next
        // terminal turn re-enters the wait via the message_end handler.
        finish({ entries: checkpointEntries(), checkpoint: true }, "checkpoint");
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
        if (allDone()) {
          finish({ entries: finalize(), checkpoint: false }, "processes:ended");
        }
      }),
      pi.events.on("processes:changed", () => {
        // A process may have ended without us catching the individual event;
        // trigger an immediate safety-net check.
        if (!settled && !timer) void tick();
      }),
    );

    if (signal?.aborted) {
      finish({ entries: finalize(), checkpoint: false }, "abort");
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });

    // One immediate check: snapshot could already be stale (process exited
    // between the initial snapshot and now).
    void tick();
  });
}

/**
 * Durable dump directory for process outputs captured when a wait ends with
 * processes still running (timeout/abort). Lives outside the run's temp log
 * dir, which @aliou deletes on session_shutdown, so the output survives for
 * forensics.
 */
const PROCESS_DUMP_ROOT = join(homedir(), ".pi", "agent", "process-dumps");

/** Guard: cap each copied log stream at this many bytes when dumping. */
const DUMP_MAX_STREAM_BYTES = 8 * 1024 * 1024;

/** Guard: inline the leading lines only for small dumps. */
const DUMP_INLINE_MAX_BYTES = 64 * 1024;
const DUMP_INLINE_LINES = 40;

/**
 * Ask the process manager for one process's on-disk log file paths (the
 * stdout/stderr/combined `.log` files under its temp log dir). Mirrors
 * `requestList`'s sync-request shape; falls back to null when the manager is
 * not loaded or does not reply.
 */
function requestLogFiles(
  pi: ExtensionAPI,
  id: string,
): Promise<{ stdoutFile: string; stderrFile: string } | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (files: unknown): void => {
      if (settled) return;
      settled = true;
      const f = files as
        | { stdoutFile?: unknown; stderrFile?: unknown }
        | null
        | undefined;
      if (
        f &&
        typeof f === "object" &&
        typeof f.stdoutFile === "string" &&
        typeof f.stderrFile === "string"
      ) {
        resolve({ stdoutFile: f.stdoutFile, stderrFile: f.stderrFile });
      } else {
        resolve(null);
      }
    };
    try {
      pi.events.emit("processes:request:log_files", {
        id,
        reply: (files: unknown) => finish(files),
      });
    } catch {
      finish(null);
    }
    setTimeout(() => finish(null), LIST_FALLBACK_MS);
  });
}

/**
 * Copy each still-running process's available log streams into a durable
 * dump directory, BEFORE @aliou's session_shutdown cleanup deletes the
 * process temp logs. Returns a list of report lines (one per dumped stream);
 * every unusual condition degrades to an explanatory note so the caller can
 * still include it in the report. Never throws.
 */
async function dumpStillRunningOutputs(
  pi: ExtensionAPI,
  entries: ReportEntry[],
): Promise<string[]> {
  const stillRunning = entries.filter(
    (e) => e.status === "still_running",
  );
  if (stillRunning.length === 0) return [];

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const dumpDir = join(PROCESS_DUMP_ROOT, stamp);
  try {
    mkdirSync(dumpDir, { recursive: true });
  } catch {
    return [
      `[process-barrier] could not create dump dir ${dumpDir}; skipped output capture`,
    ];
  }

  const notes: string[] = [];
  for (const entry of stillRunning) {
    let files: { stdoutFile: string; stderrFile: string } | null = null;
    try {
      files = await requestLogFiles(pi, entry.id);
    } catch {
      files = null;
    }
    if (!files) {
      notes.push(
        `- ${entry.id} "${entry.name}" still running; no log files available to capture`,
      );
      continue;
    }

    for (const [kind, file] of [
      ["stdout", files.stdoutFile],
      ["stderr", files.stderrFile],
    ] as const) {
      try {
        if (!existsSync(file)) continue;
        const size = statSync(file).size ?? 0;
        if (size <= 0) continue;
        const bytes = readFileSync(file);
        const capped = bytes.length > DUMP_MAX_STREAM_BYTES
          ? bytes.subarray(bytes.length - DUMP_MAX_STREAM_BYTES)
          : bytes;
        const dest = join(dumpDir, `${entry.id}.${kind}.log`);
        writeFileSync(dest, capped);
        notes.push(
          `- ${entry.id} "${entry.name}" ${kind} dumped to ${dest} (${capped.length} bytes)`,
        );
        if (capped.length <= DUMP_INLINE_MAX_BYTES) {
          const head = capped
            .toString("utf-8")
            .split(/\r?\n/)
            .filter((l) => l.length > 0)
            .slice(0, DUMP_INLINE_LINES)
            .join("\n");
          if (head.length > 0) {
            notes.push(`  head:\n${head}`);
          }
        }
      } catch {
        notes.push(
          `- ${entry.id} "${entry.name}" ${kind} could not be captured (unreadable/deleted)`,
        );
      }
    }
  }
  return notes;
}

/**
 * Resolve the durable log target for this session. Preference: alongside the
 * child's session.jsonl (async run sessions are named by run id, so the log
 * stays correlate-able with the run dir's status/events files); fallback to
 * os.tmpdir()/pi-process-barrier/session-<pid>.log for sessions without a
 * session file (e.g. in-memory sessions). Never throws.
 */
function resolveLogTarget(ctx: unknown): { path: string; where: string } {
  try {
    const sm = (
      ctx as {
        sessionManager?: { getSessionFile?: () => string | undefined } | null;
      } | null
    )?.sessionManager;
    const sessionFile = sm?.getSessionFile?.();
    if (sessionFile) {
      return {
        path: join(dirname(sessionFile), LOG_FILE_NAME),
        where: "session-dir",
      };
    }
  } catch {
    // fall through to the tmpdir fallback
  }
  return {
    path: join(tmpdir(), "pi-process-barrier", `session-${process.pid}.log`),
    where: "tmpdir-fallback",
  };
}

/**
 * Append one line, truncating oldest lines when the log grows past
 * LOG_MAX_BYTES (keeping roughly LOG_TRIM_KEEP_BYTES of the newest lines,
 * aligned to whole lines). Logging must never throw into the event pipeline:
 * every failure is swallowed and never affects wait/wake/delivery behavior.
 */
function appendLogLine(path: string, line: string): void {
  try {
    try {
      if (statSync(path).size > LOG_MAX_BYTES) {
        const old = readFileSync(path);
        const kept = old.subarray(
          Math.max(0, old.length - LOG_TRIM_KEEP_BYTES),
        );
        const nl = kept.indexOf(10); // first newline: keep whole lines only
        writeFileSync(path, nl >= 0 ? kept.subarray(nl + 1) : kept);
      }
    } catch {
      // missing/unreadable file: fall through and just append
    }
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, line + "\n");
  } catch {
    // log I/O must never break the extension
  }
}

/** Sender shape shared by handler ctx and the ExtensionAPI actions. */
type Sender = { sendUserMessage?: (content: string, options?: object) => unknown };

/**
 * Queue the report as a follow-up user message so the agent gets a reaction
 * turn. Handler ctx exposes sendUserMessage; the extension actions are the
 * fallback. Never throws (a failed delivery must not break the relay).
 * Delivery outcomes are logged: `queued` after a successful call; `FAILED`
 * with the reason on the previously-silent no-op path (no sendUserMessage on
 * ctx or pi) and on throw. (True delivery/consumption happens when the agent
 * loop processes the queue — not observable here, hence "queued".)
 */
function deliverFollowUp(
  api: ExtensionAPI,
  ctx: unknown,
  text: string,
  log?: (event: string, detail?: string) => void,
): void {
  try {
    const viaCtx = (ctx as Sender | null)?.sendUserMessage;
    if (typeof viaCtx === "function") {
      viaCtx.call(ctx, text, { deliverAs: "followUp" });
      log?.("follow-up queued", "via=ctx");
      return;
    }
    const viaPi = (api as unknown as Sender).sendUserMessage;
    if (typeof viaPi === "function") {
      viaPi.call(api, text, { deliverAs: "followUp" });
      log?.("follow-up queued", "via=pi");
      return;
    }
    // Previously silent: neither channel can queue. Now at least visible in
    // the durable log (the agent will settle without a reaction turn).
    log?.("follow-up FAILED", "reason=no sendUserMessage on ctx or pi");
    console.error("[process-barrier] no sendUserMessage on ctx or pi; follow-up dropped");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log?.("follow-up FAILED", `reason=${message}`);
    console.error(
      `[process-barrier] failed to queue follow-up: ${message}`,
    );
  }
}

export default function (pi: ExtensionAPI): void {
  // Factory-level config, read once at extension init. Precedence (highest
  // wins): pi settings key "processBarrier" -> env var -> default.
  const settings = readSettings();

  // Hard cap per wait: settings waitTimeoutSeconds -> env
  // PI_PROCESS_BARRIER_TIMEOUT_SECONDS -> default 600s; always within
  // [1, MAX_WAIT_TIMEOUT_SECONDS].
  const waitTimeoutSeconds = Math.min(
    Math.max(
      settings.waitTimeoutSeconds ??
        readEnvInt(
          "PI_PROCESS_BARRIER_TIMEOUT_SECONDS",
          DEFAULT_WAIT_TIMEOUT_SECONDS,
        ),
      1,
    ),
    MAX_WAIT_TIMEOUT_SECONDS,
  );

  // Check-in interval: settings checkIntervalSeconds (0 = disabled) -> env
  // PI_PROCESS_BARRIER_CHECK_INTERVAL_MS (ms, 0 = disabled) -> default 120s.
  // Check-ins are also the liveness signal on long holds (real LLM activity).
  let checkIntervalSeconds: number;
  if (
    settings.checkIntervalSeconds !== undefined &&
    Number.isFinite(settings.checkIntervalSeconds) &&
    settings.checkIntervalSeconds !== 0
  ) {
    checkIntervalSeconds = Math.min(
      Math.max(settings.checkIntervalSeconds, MIN_CHECK_INTERVAL_MS / 1000),
      MAX_CHECK_INTERVAL_SECONDS,
    );
  } else if (settings.checkIntervalSeconds === 0) {
    checkIntervalSeconds = 0;
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

  // Durable run log: resolved lazily on the first handler call (the session
  // file is only reachable via the handler's ctx), then reused for the
  // session's lifetime. All writes go through appendLogLine (failure-safe).
  let logTarget: { path: string; where: string } | null = null;
  const log = (event: string, detail = ""): void => {
    if (!logTarget) return;
    try {
      appendLogLine(
        logTarget.path,
        `${new Date().toISOString()} ${event}${detail ? ` ${detail}` : ""}`,
      );
    } catch {
      // logging must never throw into the event pipeline
    }
  };

  /**
   * One-line summary of a completed terminal assistant message for the gate
   * log. While our handler holds, every other mirror is blind: pi core
   * awaits extension handlers BEFORE public dispatch and session
   * persistence (agent-session.js:580->581->595), so neither events.jsonl
   * nor session.jsonl shows the completed turn until the hold ends. If a
   * run dies mid-hold, THIS line is the only durable proof the turn
   * completed normally. Never throws; degrades to a placeholder on odd
   * content shapes.
   */
  const summarizeGateMessage = (m: {
    id?: unknown;
    content?: unknown;
  }): string => {
    try {
      let totalChars = 0;
      let firstText = "";
      let firstThinking = "";
      if (typeof m.content === "string") {
        totalChars = m.content.length;
        firstText = m.content;
      } else if (Array.isArray(m.content)) {
        for (const block of m.content) {
          const b = block as {
            type?: string;
            text?: unknown;
            thinking?: unknown;
          };
          const t = typeof b?.text === "string" ? b.text : "";
          const th = typeof b?.thinking === "string" ? b.thinking : "";
          totalChars += t.length + th.length;
          if (!firstText && t) firstText = t;
          if (!firstThinking && th) firstThinking = th;
        }
      }
      const source = firstText || firstThinking;
      const preview = source
        .replace(/[\t\r\n]+/g, " ")
        .replace(/["\\]/g, "'")
        .replace(/\s{2,}/g, " ")
        .trim()
        .slice(0, 100);
      const id = typeof m.id === "string" && m.id ? m.id : "-";
      return `stopReason=stop id=${id} contentChars=${totalChars} preview="${preview}"`;
    } catch {
      return 'stopReason=stop id=- contentChars=? preview="(unavailable)"';
    }
  };

  // The single handler this extension installs. pi core awaits it before
  // relaying message_end to public listeners, so while it runs, pi-subagents
  // has not yet seen the terminal message and cannot arm the settle-drain.
  pi.on("message_end", async (event, ctx) => {
    try {
      if (!logTarget) {
        logTarget = resolveLogTarget(ctx);
        log(
          "init",
          `log=${logTarget.path} via=${logTarget.where} waitTimeoutSeconds=${waitTimeoutSeconds} checkIntervalSeconds=${checkIntervalSeconds} child=${process.env.PI_SUBAGENT_CHILD ?? "-"}`,
        );
      }
      const m = event?.message as {
        role?: string;
        stopReason?: string;
        content?: unknown[] | string | null;
      } | null;
      // Only terminal assistant messages: last model turn, no pending tools.
      if (!m || m.role !== "assistant" || m.stopReason !== "stop") return;
      // Defensive: never wait on truncated/failed/aborted turns.
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

      // Snapshot currently live processes (turn-end gate: retries a slow
      // manager within a bounded budget before concluding "nothing running").
      const { list: all, replied } = await requestListAtTurnEnd(pi);
      const running = all.filter(
        (p) => p && typeof p.id === "string" && isLive(p.status),
      );
      if (running.length === 0) {
        if (!replied) {
          // Every gate attempt timed out, so waiting is skipped on
          // inconclusive grounds; say so instead of failing silently. (A
          // prompt, definite empty reply is the healthy idle path and stays
          // quiet.)
          console.error(
            `[process-barrier] no process-list reply after ${LIST_RETRY_MAX_ATTEMPTS} attempt(s); skipping wait`,
          );
          log("gate", "live=0 skip: no manager reply (inconclusive)");
        } else {
          log("gate", "live=0 fast no-op");
        }
        return;
      }
      log(
        "gate",
        `live=${running.length} ids=${running.map((p) => `${p.id}("${p.name}")`).join(",")}`,
      );
      // Durable proof the terminal turn COMPLETED, written before our hold
      // blinds the events mirror and the session file (see the helper's
      // doc comment).
      log(
        "gate msg",
        summarizeGateMessage(m as { id?: unknown; content?: unknown }),
      );

      const signal = (
        ctx as { getSignal?: () => AbortSignal | undefined } | null | undefined
      )?.getSignal?.();
      const waitStartedAt = Date.now();
      const deadline = waitStartedAt + waitTimeoutSeconds * 1000;
      log(
        "hold start",
        `processes=${running.length} ids=${running.map((p) => p.id).join(",")} cap=${waitTimeoutSeconds}s checkin=${checkIntervalSeconds}s`,
      );
      console.error(
        `[process-barrier] holding run open for ${running.length} process(es): ${running.map((p) => p.id).join(", ")} (cap ${waitTimeoutSeconds}s, check-in ${checkIntervalSeconds}s)`,
      );

      const { entries, checkpoint, cause } = await waitForAllExited(
        pi,
        running,
        deadline,
        signal,
        checkIntervalSeconds > 0 ? checkIntervalSeconds * 1000 : 0,
      );
      log(
        "hold end",
        `cause=${cause} elapsed=${Math.max(0, Math.round((Date.now() - waitStartedAt) / 1000))}s finals=${entries.map((e) => `${e.id}:${e.status}`).join(",")}`,
      );

      // Aborted session: do not queue a follow-up; the parent owns the
      // outcome of a terminating run.
      if (signal?.aborted) {
        console.error(
          "[process-barrier] session aborted during wait; no follow-up queued",
        );
        log("follow-up skipped", "reason=session aborted");
        return;
      }

      if (checkpoint) {
        const elapsedSeconds = Math.max(
          0,
          Math.round((Date.now() - waitStartedAt) / 1000),
        );
        log("checkpoint", `queued elapsed=${elapsedSeconds}s`);
        deliverFollowUp(
          pi,
          ctx,
          `[process-barrier] ${formatCheckpoint(entries, elapsedSeconds)}`,
          log,
        );
        return;
      }

      let reportText = formatReport(
        entries,
        Math.max(0, Math.round((deadline - Date.now()) / 1000)),
      );
      // Flush-on-exit: when the wait ended with processes still running
      // (cap hit), copy their logs to a durable dump dir before @aliou's
      // session_shutdown cleanup deletes them.
      if (entries.some((e) => e.status === "still_running")) {
        try {
          const dumpNotes = await dumpStillRunningOutputs(pi, entries);
          if (dumpNotes.length > 0) {
            reportText +=
              "\n\n[process-barrier] output captured for still-running processes:\n" +
              dumpNotes.join("\n");
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          reportText +=
            `\n\n[process-barrier] output capture failed: ${message}`;
        }
      }
      const stillRunningCount = entries.filter(
        (e) => e.status === "still_running",
      ).length;
      log("follow-up queued", `kind=report stillRunning=${stillRunningCount}`);
      deliverFollowUp(
        pi,
        ctx,
        `[process-barrier] Background process wait finished:\n${reportText}\nReport on these results in your final answer.`,
        log,
      );
    } catch (err) {
      // Never throw from a message_end handler: a failure here must not
      // break the turn or the relay.
      const message = err instanceof Error ? err.message : String(err);
      log("handler error", message);
      console.error(`[process-barrier] error: ${message}`);
    }
  });
}
