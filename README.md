# pi-process-barrier

A small [pi](https://github.com/earendil-works/pi) extension that keeps pi-subagents child runs alive while spawned processes (managed by the `@aliou/pi-processes` extension) are still running, and wakes the agent with a reaction turn when they finish.

## What it does

- Registers a blocking tool named `barrier` **eagerly at session start** (required for child-session spawn preflight and by-name tool-call resolution) but **inactive by default**: the tool is present in the model's callable schema only while processes are actually running (see [Dynamic availability](#dynamic-availability)).
- Hooks `message_end`: when an assistant turn ends while processes are still running, it **injects a synthetic `toolCall` for `barrier` into the final message** (in place).
- The barrier's `execute()` **blocks** until every watched process exits (or the timeout passes), then returns a summary of each process's final status as its `tool_result`.
- The agent loop's automatic follow-up LLM call is the **reaction turn**: with the summary in context, the agent reports, retries, or starts more processes. When it finally ends its turn with nothing running, the run completes normally.
- When **no processes are running** the tool is **absent from the model's function-calling schema** (see [Dynamic availability](#dynamic-availability)), so models cannot call it ad hoc at all.
- If a direct call somehow still happens with **no processes running**, `execute()` **throws** an internal-helper error — kept as a safety net.
- Returns **no `terminate: true`** (the follow-up LLM call is required for the reaction).
- Has **no `promptSnippet`**, so it is omitted from the "Available tools" section of the default system prompt — quiet unless auto-injected (and active).

## Dynamic availability

The tool is registered for the whole session but only *visible* while needed:

- **Registered at session start (eagerly)**: `pi.registerTool()` runs once at factory init. Registration must happen before any subagent spawns: pi-subagents validates each agent's frontmatter `tools:` allowlist against *registered* tools, and agents that list `barrier` (`delegate`, `oracle`, `scout`, `worker`, `worker-full`) would fail to spawn with "requested unavailable child tools" without it. Injected `toolCall`s also resolve by name only while the tool is registered.
- **Active-set toggle**: the extension reads `pi.getActiveTools()` and calls `pi.setActiveTools()` to add/remove `barrier`. Tools outside the active set are excluded from the model's schema — this is what hides the tool while idle.
- The toggle is driven purely by **live-process state** (`running`/`terminating`): session-scope listeners on the manager's `processes:changed`/`processes:ended` broadcasts (with an authoritative `processes:request:list` snapshot as fallback) activate the tool while any process is live and hide it once none are; the `message_end` handler re-asserts the same from its own snapshot, and `execute()` re-hides it in a `finally` when a wait ends.
- **While idle the tool is invisible to models** (absent from the model's callable schema / system prompt), so models cannot call `barrier` ad hoc; the direct-call guard remains purely as a safety net.

## History scrubbing

Deactivation keeps the tool out of the model's schema, but a model that once *saw* a barrier call in its history (the auto-injected `toolCall` + its `toolResult`) could still imitate it. To close that gap the extension scrubs retired (and blocked) invocations from the model-visible context:

- The current invocation (the pair of the injected `toolCall` and its `toolResult`) **stays visible** to the immediate wake-up follow-up LLM call by design — that call reads the process summary from the `toolResult` to decide what to report next. Showing it in that context projection **serves** it.
- It is **retired as soon as that wake-up call has happened**: the moment the wake-up call's assistant message is finalized (`message_end` fires for every persisted assistant message, including ones carrying further tool calls), the pair moves to the retired set. Retirement never waits for the next user turn or the next barrier, so **any later LLM call in the same turn** (e.g. the wake-up response triggered more tool calls) or in later turns never sees it — the model cannot imitate the tool from history.
- A `message_start` with role `user` and a new injected barrier call superseding a previous one (only one barrier runs at a time) remain as **idempotent safety nets** for retiring a pair that was never served.
- On every subsequent LLM call the extension returns a modified context (via the pi `context` event, which fires before every LLM call with a deep copy of the exact messages about to be sent): the retired `toolResult` message is dropped, and the retired `toolCall` block is stripped from its assistant message (the whole message is dropped if it had only toolCall blocks). The model therefore never sees a past barrier invocation in history and cannot imitate it.
- The **stored session file still contains the pair**: pi sessions are append-only and there is no entry-deletion API (no hidden flag on tool calls/results), so scrubbing is per-LLM-call, applied to the context copy only.
- The handler returns `undefined` (no-op) when nothing is retired/current and never throws, so it can never break an LLM call.

## Call provenance

The extension also blocks **model-authored** `barrier` calls before they can execute, so even an attempted call produces no real barrier wait and no history entry the model sees:

- The pi `tool_call` event fires before every tool executes and can block a call with `{ block: true, reason }`. The extension records the id of every call it **injects** (`injectedBarrierCallIds`) and allows only those. Any other `barrier` call id is model-authored and is **blocked unconditionally**, with the same internal-helper message the direct-call guard uses — whether or not an injected wait is currently in flight. The tool's `execute()` never runs, so no wait starts and no duplicate watch is created.
- A blocked call still surfaces as an error `toolResult` in the stored session (the agent loop preflights calls and turns a block into an immediate error result). The extension records blocked ids (`blockedBarrierCallIds`) on `tool_result` and the context scrub hides them too, so **the model never sees its own attempted call in the messages sent to it** — no result, no history entry.
- **No legitimate unrecorded case exists**: registration and injection are per-session, and an injected id is always recorded in `injectedBarrierCallIds` *before* the agent loop executes that call — so an unrecorded id is by definition model-authored and is always blocked. The `execute()` direct-call guard remains only as a last-resort safety net.

## Mechanism

- pi-subagents children are in-process `AgentSession`s (`mode: "print"`). The parent only computes a terminal stop from the **relayed** child `message_end`, and only arms its settle-drain (`1s` grace → abort → `3s` hard finish → `dispose` → `session_shutdown` → `@aliou`'s cleanup `killAll()`) when the final assistant message has **no tool calls**.
- The `message_end` extension handler runs **before** the relay, so injecting a toolCall makes the parent see `hasToolCall = true` → the drain **never arms** → no abort, no dispose, no `killAll()` → child and processes survive.
- Tool calls resolve **by name with no provenance check**, so the injected call executes like any normal call.
- **Visibility is separate from resolvability**: the tool is added to the *active tool set* (what the model sees) only while processes run, but injected calls resolve by name as long as the tool is registered and the agent's `tools:` allowlist keeps `barrier` listed.
- `execute()` blocks using the shared extension event bus: `processes:request:list` to snapshot/refresh, `processes:ended` (and `processes:changed`) broadcast notifications, a 2.5s polling safety net, and a timeout defaulting to 600s (max 1800s).
- On completion the loop emits the automatic follow-up LLM call — that **is** the wake-up reaction.
- If a configurable check-in interval elapses while processes still run, the barrier resolves early with a *checkpoint* result (no `terminate: true`), the agent is woken to verify nothing is hanging, and when it ends its turn the barrier is re-injected for a fresh wait (resolve-and-reinject). Exactly one LLM call per check-in; the drain can never arm anywhere in the cycle.

## Configuration

The periodic check-in interval (waking the agent to check whether processes are hanging) is configured in **pi settings** at `/Users/mimi/.pi/agent/settings.json` under a top-level `processBarrier` key:

```jsonc
{
  "processBarrier": {
    "checkIntervalSeconds": 300   // 0 = disable periodic check-ins; default 300
  }
}
```

Precedence (highest wins):
1. Direct tool call param `checkIntervalSeconds` (manual calls only)
2. Pi settings `processBarrier.checkIntervalSeconds` (0 disables; clamped to 5s–24h)
3. Env `PI_PROCESS_BARRIER_CHECK_INTERVAL_MS` (0 disables; default 300000)
4. Default: 300s

`0` disables periodic check-ins entirely — the barrier then only resolves on all-processes-exit or the timeout.

pi exposes no settings API to extensions; the extension reads the settings.json file at factory init, following the same pattern pi-subagents uses for its `subagents` key. A reload is required for settings or code changes to take effect.

## Install / wiring

- The extension lives in its own git repo: `~/Projects/pi-process-barrier/process-barrier.ts`.
- It is symlinked into pi's extension dir: `/Users/mimi/.pi/agent/extensions/process-barrier.ts -> /Users/mimi/Projects/pi-process-barrier/process-barrier.ts`, so edits in the repo take effect on the next pi reload/restart (no config change needed).
- Five agents have the extension path in their frontmatter `extensions:` line **and** `barrier` in their `tools:` allowlist: `delegate`, `oracle`, `scout`, `worker`, `worker-full`. **Keep `barrier` listed**: resolvability is governed by the allowlist (an unlisted injected tool resolves to "Tool not found" and the run would settle), while *visibility* is governed by the active-set toggle in [Dynamic availability](#dynamic-availability) — the tool stays hidden from the schema while idle, but must remain resolvable for injected calls while processes run.
- The main agent is enabled via `parent-tools-lockdown.ts`, which now lists **only `subagent`** (`setActiveTools(["subagent"])`); `barrier` is no longer hard-listed there (separate change) — the parent has no `process` tool anyway, and the barrier manages its own visibility.
- `reviewer.md` is intentionally excluded — it has no `process` tool, so processes cannot run there.
- Restart/reload pi after editing the extension for new sessions to pick it up.

## Known caveats

- While processes are running the tool is deliberately present in the model's schema (injected calls must be callable). It has no `promptSnippet`, and the direct-call guard remains as a safety net for any call with nothing to wait for.
- `terminate_timeout` counts as "done" for reporting.
- Only one barrier runs at a time (single injected toolCall per final message).
- Avoid setting a global `toolTimeoutMs` that would hard-kill a long `wait`' the run-level `timeoutMs` still bounds the wait.

## Development

```console
cd ~/Projects/pi-process-barrier
# edit process-barrier.ts, then reload pi (e.g. /reload) to pick it up in new sessions
```