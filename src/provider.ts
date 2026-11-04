import type { Api, AssistantMessage, AssistantMessageEventStream, Model, SimpleStreamOptions, StopReason, TextContent, ThinkingContent, ToolCall, ToolResultMessage, TranscriptContext } from "@earendil-works/pi-ai";
import { calculateCost, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { AnchorRegistry } from "./anchors.ts";
import { fingerprint, historyText, type InputPart, piToolName, readTranscript, type Transcript, turnInput } from "./convert.ts";
import type { HostSpec, MuseSession, ToolCallInfo } from "./host.ts";
import { museEffort } from "./models.ts";
import { errorText, sha } from "./util.ts";

export const API = "muse-code-sdk";

export interface ProviderOptions {
	registry: AnchorRegistry;
	/** Starts a Muse Code session (MuseHost.open in production; tests substitute a fake). */
	openSession: (spec: HostSpec) => Promise<MuseSession>;
	/** Live Muse hosts kept for reuse (one per concurrent conversation). */
	maxSessions?: number;
	/** Close a host idle this long; the conversation resumes from Muse Code's session on its next turn. */
	idleMs?: number;
	debug?: (msg: string) => void;
	cwd?: () => string;
}

/** How a request was attached to Muse Code; exposed for tests and the debug log. */
export type Route = "live" | "resume" | "fork" | "synth" | "fresh";

type StepResult = { ok: true; turnEnded: boolean; completed: boolean } | "aborted" | { ok: false; error: string };

export class MuseSdkProvider {
	private sessions: MuseSession[] = [];
	/** Every host whose process may still run, pooled or not: the leak backstop closes from here. */
	private readonly all = new Set<MuseSession>();
	/** Hosts a provider call is reading right now. */
	private readonly active = new Set<MuseSession>();
	/** Provider calls in flight, including one still opening its host. */
	private inFlight = 0;
	/** Set by `retire()`: hosts drop as their call finishes instead of returning to the pool. */
	private retired = false;
	private onDrained: (() => void) | undefined;
	private readonly opts: Required<Omit<ProviderOptions, "debug">> & Pick<ProviderOptions, "debug">;
	private timer?: NodeJS.Timeout;
	lastRoute?: Route;

	constructor(opts: ProviderOptions) {
		const given = Object.fromEntries(Object.entries(opts).filter(([, v]) => v !== undefined)) as ProviderOptions;
		this.opts = { maxSessions: 3, idleMs: 20 * 60_000, cwd: () => process.cwd(), ...given };
	}

	readonly streamSimple = (model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions): AssistantMessageEventStream => {
		const stream = createAssistantMessageEventStream();
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			stopReason: "pending",
			timestamp: Date.now(),
		};
		void this.run(model, context, options, stream, output);
		return stream;
	};

	private log(msg: string): void {
		this.opts.debug?.(msg);
	}

	private async run(model: Model<Api>, context: TranscriptContext, options: SimpleStreamOptions | undefined, stream: AssistantMessageEventStream, output: AssistantMessage): Promise<void> {
		let session: MuseSession | undefined;
		let started = false;
		const fail = (reason: "error" | "aborted", message: string) => {
			output.stopReason = reason;
			output.errorMessage = message;
			if (!started) stream.push({ type: "start", partial: output });
			stream.push({ type: "error", reason, error: output });
			stream.end();
		};
		this.inFlight += 1;
		try {
			if (options?.maxTokens === 1) {
				// A cache-warming replay. Muse Code keeps its own prompt cache per session; answer without
				// touching a live session (a replay of an older context would otherwise end it).
				output.stopReason = "length";
				stream.push({ type: "start", partial: output });
				stream.push({ type: "done", reason: "length", message: output });
				stream.end();
				return;
			}
			const t = readTranscript(context);
			if (t.messages.length === 0 || t.messages[t.messages.length - 1].role === "assistant") {
				return fail("error", "pi-muse-subscription: the transcript must end with a user message or tool results");
			}
			const effort = museEffort(options?.reasoning, (model as { thinkingLevelMap?: Record<string, string | null> }).thinkingLevelMap);
			const spec: Omit<HostSpec, "open"> = {
				cwd: this.opts.cwd(),
				model: model.id,
				systemPrompt: t.systemPrompt,
				tools: t.tools,
				toolsHash: t.toolsHash,
				ephemeral: t.tools.length === 0,
			};

			const attached = await this.attach(t, spec);
			session = attached.session;
			this.lastRoute = attached.route;
			this.log(`route=${attached.route} session=${session.sessionId} prefix=${attached.prefix}/${t.messages.length}`);
			await options?.onPayload?.({ sessionId: session.sessionId, route: attached.route, model: model.id, tail: t.messages.length - attached.prefix }, model);

			// Hand pi's new messages to Muse Code.
			const tail = t.messages.slice(attached.prefix);
			const results = tail.filter((m): m is ToolResultMessage => m.role === "toolResult");
			if (attached.route === "live" && session.awaiting.size > 0) {
				// Mid-turn: a message typed while tools ran is steered in before the parked calls are
				// released, so the model sees it at this tool boundary.
				const extra = turnInput(tail.filter((m) => m.role !== "toolResult"));
				if (extra.length > 0) await session.steer(extra);
				for (const r of results) session.deliver(r);
			} else {
				const input: InputPart[] = [...(attached.history ? [{ type: "text" as const, text: attached.history }] : []), ...turnInput(tail)];
				if (!spec.ephemeral) this.opts.registry.openTurn(session.sessionId);
				await session.startTurn(input, effort);
			}
			session.chain = t.chain[t.messages.length];
			session.count = t.messages.length;

			const step = await this.readStep(session, model, options, stream, output, () => {
				started = true;
			});
			if (step === "aborted") return fail("aborted", "Request was aborted");
			if (!step.ok) {
				// Muse Code is in an unknown state after a failure; the next call resumes or forks from its record.
				this.drop(session);
				return fail("error", step.error);
			}

			session.chain = sha(`${session.chain}\0${fingerprint(output)}`);
			session.count += 1;
			if (step.turnEnded && step.completed && !spec.ephemeral && session.turnId) {
				// Muse Code can fork or resume a session only at a completed turn: record those.
				this.opts.registry.put(session.chain, { sessionId: session.sessionId, turnId: session.turnId, cwd: spec.cwd });
			}
			session.awaiting = new Set(output.content.filter((c): c is ToolCall => c.type === "toolCall").map((c) => c.id));
			if (output.stopReason !== "toolUse") session.awaiting.clear();
			stream.push({ type: "done", reason: output.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">, message: output });
			stream.end();
			if (spec.ephemeral) this.drop(session);
		} catch (e) {
			if (session) this.drop(session);
			fail(options?.signal?.aborted ? "aborted" : "error", errorText(e, session?.stderr));
		} finally {
			if (session) {
				session.busy = false;
				this.active.delete(session);
				// A retiring provider keeps no pool: this host's call is done, so it drops now.
				if (this.retired && !session.dead) this.drop(session);
			}
			this.inFlight -= 1;
			this.checkDrained();
			if (!this.retired) this.scheduleSweep();
		}
	}

	/**
	 * Find or start the Muse Code session for this transcript. In order of preference: a live host
	 * that has consumed a prefix of it; Muse Code's own session at the longest prefix ending in a
	 * completed turn (resumed, or forked when that turn is not the session's last); a fresh session
	 * whose first turn carries the earlier conversation as text; a fresh one.
	 */
	private async attach(t: Transcript, spec: Omit<HostSpec, "open">): Promise<{ session: MuseSession; prefix: number; route: Route; history?: string }> {
		const n = t.messages.length;
		for (const s of this.sessions) {
			if (s.dead || s.busy || s.spec.ephemeral !== spec.ephemeral || s.spec.systemPrompt !== t.systemPrompt || s.spec.toolsHash !== t.toolsHash || s.spec.cwd !== spec.cwd) continue;
			if (s.count > n || t.chain[s.count] !== s.chain) continue;
			const tail = t.messages.slice(s.count);
			if (tail.some((m) => m.role === "assistant")) continue;
			const resultIds = new Set(tail.filter((m): m is ToolResultMessage => m.role === "toolResult").map((m) => m.toolCallId));
			if (s.awaiting.size > 0 ? ![...s.awaiting].every((id) => resultIds.has(id)) : resultIds.size > 0) continue;
			s.busy = true; // before any await, so a concurrent call cannot select it too
			this.active.add(s);
			s.lastUsed = Date.now();
			if (s.model !== spec.model && s.awaiting.size === 0) {
				try {
					await s.setModel(spec.model);
				} catch (e) {
					s.busy = false;
					this.active.delete(s);
					this.drop(s);
					throw e;
				}
			}
			return { session: s, prefix: s.count, route: "live" };
		}

		if (!spec.ephemeral) {
			// Longest prefix ending in an assistant message whose Muse Code turn completed.
			for (let k = n - 1; k >= 1; k--) {
				if (t.messages[k - 1].role !== "assistant") continue;
				const a = this.opts.registry.get(t.chain[k]);
				if (!a || a.cwd !== spec.cwd) continue;
				// Another call is mid-response on that session; taking it over would end that call.
				if (this.sessions.some((s) => s.busy && s.sessionId === a.sessionId)) continue;
				// A live host on that session is stale by now; it must be gone before the session is reopened.
				for (const s of this.sessions) if (s.sessionId === a.sessionId) this.drop(s);
				await this.waitExited(a.sessionId);
				const atTail = this.opts.registry.tail(a.sessionId) === a.turnId;
				const route: Route = atTail ? "resume" : "fork";
				try {
					const session = await this.open({ ...spec, open: atTail ? { kind: "resume", sessionId: a.sessionId } : { kind: "fork", sessionId: a.sessionId, turnId: a.turnId } }, t.chain[k], k);
					return { session, prefix: k, route };
				} catch (e) {
					this.log(`${route} ${a.sessionId}@${a.turnId} failed (${errorText(e)}); trying a shorter prefix`);
				}
			}
		}

		let last = -1;
		t.messages.forEach((m, i) => {
			if (m.role === "assistant") last = i;
		});
		if (last >= 0) {
			const session = await this.open({ ...spec, open: { kind: "fresh" } }, t.chain[last + 1], last + 1);
			return { session, prefix: last + 1, route: "synth", history: historyText(t.messages.slice(0, last + 1)) };
		}
		const session = await this.open({ ...spec, open: { kind: "fresh" } }, t.chain[0], 0);
		return { session, prefix: 0, route: "fresh" };
	}

	private async open(spec: HostSpec, chain: string, count: number): Promise<MuseSession> {
		const session = await this.opts.openSession(spec);
		session.chain = chain;
		session.count = count;
		session.busy = true;
		this.active.add(session);
		this.sessions.push(session);
		this.track(session);
		this.evict();
		return session;
	}

	/** Consume Muse Code events until the current model response hands control to pi or the turn ends. */
	private async readStep(
		session: MuseSession,
		model: Model<Api>,
		options: SimpleStreamOptions | undefined,
		stream: AssistantMessageEventStream,
		output: AssistantMessage,
		onStart: () => void,
	): Promise<StepResult> {
		const signal = options?.signal;
		const blocks = output.content as (TextContent | ThinkingContent | ToolCall)[];
		const byItem = new Map<string, number>();
		const parts = new Map<string, number>();
		let started = false;
		let finishReason: string | undefined;
		const sum = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
		const start = () => {
			if (started) return;
			started = true;
			onStart();
			stream.push({ type: "start", partial: output });
		};
		const block = (itemId: string, kind: "text" | "thinking"): number => {
			const known = byItem.get(itemId);
			if (known !== undefined) return known;
			start();
			blocks.push(kind === "text" ? { type: "text", text: "" } : { type: "thinking", thinking: "" });
			const i = blocks.length - 1;
			byItem.set(itemId, i);
			stream.push({ type: kind === "text" ? "text_start" : "thinking_start", contentIndex: i, partial: output });
			return i;
		};
		const announce = (calls: ToolCallInfo[]): boolean => {
			let any = false;
			for (const c of calls) {
				const name = piToolName(c.name);
				if (!name || session.announced.has(c.callId)) continue;
				session.announced.add(c.callId);
				start();
				let args: ToolCall["arguments"] = {};
				try {
					args = c.args ? JSON.parse(c.args) : {};
				} catch {}
				const call: ToolCall = { type: "toolCall", id: c.callId, name, arguments: args };
				blocks.push(call);
				const i = blocks.length - 1;
				stream.push({ type: "toolcall_start", contentIndex: i, partial: output });
				stream.push({ type: "toolcall_delta", contentIndex: i, delta: c.args, partial: output });
				stream.push({ type: "toolcall_end", contentIndex: i, toolCall: call, partial: output });
				any = true;
			}
			return any;
		};
		const onAbort = () => void session.interrupt();
		signal?.addEventListener("abort", onAbort, { once: true });
		try {
			for (;;) {
				let ev;
				try {
					ev = await session.events.next(signal);
					session.lastUsed = Date.now();
				} catch {
					this.drop(session);
					return "aborted";
				}
				if (ev === undefined) return { ok: false, error: errorText(new Error("Muse Code exited unexpectedly"), session.stderr) };
				switch (ev.type) {
					case "text_start":
						block(ev.itemId, "text");
						break;
					case "text_delta": {
						const i = block(ev.itemId, "text");
						(blocks[i] as TextContent).text += ev.delta;
						stream.push({ type: "text_delta", contentIndex: i, delta: ev.delta, partial: output });
						break;
					}
					case "text_end": {
						const i = block(ev.itemId, "text");
						const b = blocks[i] as TextContent;
						if (ev.text.length > b.text.length && ev.text.startsWith(b.text)) {
							const rest = ev.text.slice(b.text.length);
							b.text = ev.text;
							stream.push({ type: "text_delta", contentIndex: i, delta: rest, partial: output });
						}
						stream.push({ type: "text_end", contentIndex: i, content: b.text, partial: output });
						byItem.delete(ev.itemId);
						break;
					}
					case "thinking_start":
						block(ev.itemId, "thinking");
						break;
					case "thinking_delta": {
						const i = block(ev.itemId, "thinking");
						const b = blocks[i] as ThinkingContent;
						const prev = parts.get(ev.itemId);
						const delta = prev !== undefined && prev !== ev.part && b.thinking ? `\n\n${ev.delta}` : ev.delta;
						parts.set(ev.itemId, ev.part);
						b.thinking += delta;
						stream.push({ type: "thinking_delta", contentIndex: i, delta, partial: output });
						break;
					}
					case "thinking_end": {
						const i = block(ev.itemId, "thinking");
						const b = blocks[i] as ThinkingContent;
						if (!b.thinking && ev.text) {
							b.thinking = ev.text;
							stream.push({ type: "thinking_delta", contentIndex: i, delta: ev.text, partial: output });
						}
						stream.push({ type: "thinking_end", contentIndex: i, content: b.thinking, partial: output });
						byItem.delete(ev.itemId);
						break;
					}
					case "usage": {
						finishReason = ev.finishReason;
						sum.input += Math.max(0, ev.promptTokens - ev.cacheRead - ev.cacheWrite);
						sum.output += ev.outputTokens;
						sum.cacheRead += ev.cacheRead;
						sum.cacheWrite += ev.cacheWrite;
						Object.assign(output.usage, sum);
						// pi reads totalTokens as the context size: that of the latest response, not a sum.
						output.usage.totalTokens = ev.promptTokens + ev.outputTokens;
						output.usage.cost = calculateCost(model, output.usage);
						if (ev.modelId) output.responseModel = ev.modelId;
						break;
					}
					case "tool_calls":
						if (announce(ev.calls)) {
							output.stopReason = "toolUse";
							return { ok: true, turnEnded: false, completed: false };
						}
						break;
					case "tool_started":
						// Without the raw log, calls surface one at a time as Muse Code starts them.
						if (announce([ev.call])) {
							output.stopReason = "toolUse";
							return { ok: true, turnEnded: false, completed: false };
						}
						break;
					case "turn_completed":
						if (ev.terminal === "completed") {
							start();
							output.stopReason = finishReason === "length" || finishReason === "max_output_tokens" ? "length" : "stop";
							return { ok: true, turnEnded: true, completed: true };
						}
						if (ev.terminal === "cancelled") return signal?.aborted ? "aborted" : { ok: false, error: "Muse Code cancelled the turn" };
						return { ok: false, error: turnErrorText(ev.error, ev.reason) };
					case "host_error":
						return { ok: false, error: ev.message };
				}
			}
		} finally {
			signal?.removeEventListener("abort", onAbort);
		}
	}

	private closing = new Map<string, Promise<void>>();

	private drop(s: MuseSession): void {
		s.close();
		this.sessions = this.sessions.filter((x) => x !== s);
		if (!s.sessionId) return;
		const exited = s.exited.finally(() => {
			if (this.closing.get(s.sessionId) === exited) this.closing.delete(s.sessionId);
		});
		this.closing.set(s.sessionId, exited);
	}

	private async waitExited(sessionId: string): Promise<void> {
		const p = this.closing.get(sessionId);
		if (p) await Promise.race([p, new Promise((r) => setTimeout(r, 10_000).unref())]);
	}

	private track(s: MuseSession): void {
		this.all.add(s);
		void s.exited.then(() => this.all.delete(s));
	}

	private evict(): void {
		this.sessions = this.sessions.filter((s) => !s.dead);
		// Hosts in a provider call are never evicted, so the pool may briefly exceed maxSessions.
		const idle = this.sessions.filter((s) => !this.active.has(s)).sort((a, b) => a.lastUsed - b.lastUsed);
		while (this.sessions.length > this.opts.maxSessions && idle.length > 0) this.drop(idle.shift()!);
		// Backstop: a host that left the pool without being closed is closed once more than
		// maxSessions processes run outside provider calls.
		const pooled = (s: MuseSession) => (this.sessions.includes(s) ? 1 : 0);
		const stray = [...this.all].filter((s) => !s.dead && !this.active.has(s)).sort((a, b) => pooled(a) - pooled(b) || a.lastUsed - b.lastUsed);
		while (stray.length > this.opts.maxSessions) this.drop(stray.shift()!);
	}

	private scheduleSweep(): void {
		if (this.timer) return;
		this.timer = setInterval(() => {
			this.sweep();
			if (this.all.size === 0 && this.timer) {
				clearInterval(this.timer);
				this.timer = undefined;
			}
		}, 60_000);
		this.timer.unref();
	}

	/** Close hosts idle longer than idleMs. */
	sweep(now = Date.now()): void {
		for (const s of [...this.all]) {
			if (now - s.lastUsed <= this.opts.idleMs) continue;
			// A pooled host parked on tool calls waits for pi's results (a long build, say).
			if (!this.active.has(s) && this.sessions.includes(s) && s.awaiting.size > 0) continue;
			this.drop(s);
		}
	}

	/**
	 * Graceful teardown once no runtime holds this provider any more: idle hosts (including
	 * one parked on tool calls between steps) drop now, a host with a call in flight drops
	 * when that call finishes, and `onDrained` runs once no call is in flight any more so
	 * the caller can close the bridge. An in-flight call is never killed here.
	 */
	retire(onDrained: () => void): void {
		if (this.retired) return;
		this.retired = true;
		this.onDrained = onDrained;
		for (const s of new Set([...this.sessions, ...this.all])) if (!this.active.has(s)) this.drop(s);
		this.checkDrained();
	}

	private checkDrained(): void {
		if (!this.retired || this.inFlight > 0 || !this.onDrained) return;
		const onDrained = this.onDrained;
		this.onDrained = undefined;
		onDrained();
	}

	shutdown(): void {
		this.onDrained = undefined;
		for (const s of new Set([...this.sessions, ...this.all])) this.drop(s);
	}
}

function turnErrorText(error: { kind?: string; message?: string } | undefined, reason: string | undefined): string {
	if (error?.kind === "authRequired") {
		return "Muse Code is not signed in. Run `muse` in a terminal and sign in with /login (your Muse Code subscription), then retry.";
	}
	return [error?.message, reason].filter(Boolean).join(" ") || `Muse Code turn failed${error?.kind ? ` (${error.kind})` : ""}`;
}
