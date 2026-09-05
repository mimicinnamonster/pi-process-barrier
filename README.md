# pi-process-barrier

A small [pi](https://github.com/earendil-works/pi) extension that keeps pi-subagents child runs alive while spawned processes (managed by the `@aliou/pi-processes` extension) are still running, and wakes the agent with a reaction turn when they finish.

## What it does

- Registers a blocking tool named `process_barrier`.
- Hooks `message_end`: when an assistant turn ends while processes are still running, it **injects a synthetic `toolCall` for `process_barrier` into the final message** (in place).
- The barrier's `execute()` **blocks** until every watched process exits (or the timeout passes), then returns a summary of each process's final status as its `tool_result`.
- The agent loop's automatic follow-up LLM call is the **reaction turn**: with the summary in context, the agent reports, retries, or starts more processes. When it finally ends its turn with nothing running, the run completes normally.
- If an agent calls `process_barrier` directly while **no processes are running**, `execute()` **throws** an internal-helper error so the LLM learns not to call it ad hoc.
- Returns **no `terminate: true`** (the follow-up LLM call is required for the reaction).
- Has **no `promptSnippet`**, so it is omitted from the "Available tools" section of the default system prompt — quiet unless auto-injected.

## Mechanism

- pi-subagents children are in-process `AgentSession`s (`mode: "print"`). The parent only computes a terminal stop from the **relayed** child `message_end`, and only arms its settle-drain (`1s` grace → abort → `3s` hard finish → `dispose` → `session_shutdown` → `@aliou`'s cleanup `killAll()`) when the final assistant message has **no tool calls**.
- The `message_end` extension handler runs **before** the relay, so injecting a toolCall makes the parent see `hasToolCall = true` → the drain **never arms** → no abort, no dispose, no `killAll()` → child and processes survive.
- Tool calls resolve **by name with no provenance check**, so the injected call executes like any normal call.
- `execute()` blocks using the shared extension event bus: `processes:request:list` to snapshot/refresh, `processes:ended` (and `processes:changed`) broadcast notifications, a 2.5s polling safety net, and a timeout defaulting to 600s (max 1800s).
- On completion the loop emits the automatic follow-up LLM call — that **is** the wake-up reaction.

## Install / wiring

- The extension lives in its own git repo: `~/Projects/pi-process-barrier/process-barrier.ts`.
- It is symlinked into pi's extension dir: `/Users/mimi/.pi/agent/extensions/process-barrier.ts -> /Users/mimi/Projects/pi-process-barrier/process-barrier.ts`, so edits in the repo take effect on the next pi reload/restart (no config change needed).
- Five agents have the extension path in their frontmatter `extensions:` line **and** `process_barrier` in their `tools:` allowlist: `delegate`, `oracle`, `scout`, `worker`, `worker-full`. (The allowlist matters: an unlisted injected tool resolves to "Tool not found" and the run would settle.)
- The main agent is enabled via `parent-tools-lockdown.ts`: `setActiveTools(["subagent", "process_barrier"])`.
- `reviewer.md` is intentionally excluded — it has no `process` tool, so processes cannot run there.
- Restart/reload pi after editing the extension for new sessions to pick it up.

## Known caveats

- The tool is LLM-visible in the registered-tools list (it must be registered to resolve injected calls) but omissed from the available-tools prompt section and errors when called with nothing to wait for.
- `terminate_timeout` counts as "done" for reporting.
- Only one barrier runs at a time (single injected toolCall per final message).
- Avoid setting a global `toolTimeoutMs` that would hard-kill a long `wait`' the run-level `timeoutMs` still bounds the wait.

## Development

```console
cd ~/Projects/pi-process-barrier
# edit process-barrier.ts, then reload pi (e.g. /reload) to pick it up in new sessions
```