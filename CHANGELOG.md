# Changelog

## Unreleased

- Fixed a subagent's session shutdown killing the main session's in-flight turn when both run in
  one process (pi-bolt subagents): the provider and MCP bridge are now shared by every session in
  the process, and each extension runtime holds a share from the moment its factory loads (not
  from `session_start`, which a served call may never wait for). Only the last runtime's
  `session_shutdown` retires the shared provider — idle hosts drop then, a host with a call still
  in flight drops when that call finishes, and the bridge closes after it — while process exit
  still closes everything at once. A retired generation is never reused: the next factory load
  rebuilds the provider, bridge and host environment from its own config (Contributor mode
  included). Previously any session's `session_shutdown` closed every Muse host
  (`muse serve: stdin_closed`).

## 0.1.0 (2026-10-30)

Initial release.

- Muse Spark and Muse Wonder models in pi through the Muse Code SDK and `muse serve`, on the Muse
  Code subscription login. Models come from Muse Code's live catalog for your plan.
- pi's system prompt and tools replace Muse Code's; Muse Code's observer agents, skill catalog and
  foreign rules are off, so a turn is one model request per step.
- pi's tools run through a loopback MCP bridge with parked calls; all calls of a response reach pi
  at once, and a message typed during a tool run is steered into the same turn.
- Divergence (abort, `/tree`, fork, restart) resumes or forks Muse Code's own session at the last
  completed turn; foreign history starts a fresh session carrying it as text.
- `META_API_KEY` is never passed on; Contributor mode follows your saved Muse Code setting, else off.
