import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import { normalizeContext, Type, type Api, type Message, type Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ProviderConfig } from "@earendil-works/pi-coding-agent";
import { AnchorRegistry } from "../src/anchors.ts";
import { McpBridge } from "../src/bridge.ts";
import { resolveConfig } from "../src/config.ts";
import { historyText, piToolName, readTranscript, turnInput } from "../src/convert.ts";
import { cleanEnv, hostSettings } from "../src/home.ts";
import type { HostEnv, HostEvent, HostSpec, MuseSession } from "../src/host.ts";
import createMuseExtension, { type MuseExtensionDeps } from "../src/index.ts";
import { SharedLifecycle, sharedMuse } from "../src/lifecycle.ts";
import { museEffort, museModels } from "../src/models.ts";
import { API, MuseSdkProvider } from "../src/provider.ts";
import { AsyncQueue } from "../src/util.ts";

const user = (text: string): Message => ({ role: "user", content: text, timestamp: 1 });

describe("convert", () => {
	it("maps Muse tool names back to pi's", () => {
		assert.equal(piToolName("mcp__pi__read"), "read");
		assert.equal(piToolName("mcp__pi.bash"), "bash");
		assert.equal(piToolName("muse.read_file"), undefined);
	});

	it("chains transcript states by content, not timestamps", () => {
		const a = readTranscript(normalizeContext({ systemPrompt: "S", tools: [], messages: [user("hi")] }));
		const b = readTranscript(normalizeContext({ systemPrompt: "S", tools: [], messages: [{ ...user("hi"), timestamp: 99 }] }));
		const c = readTranscript(normalizeContext({ systemPrompt: "T", tools: [], messages: [user("hi")] }));
		assert.deepEqual(a.chain, b.chain);
		assert.notEqual(a.chain[1], c.chain[1]);
	});

	it("turns user content into Muse input parts", () => {
		const parts = turnInput([{ role: "user", content: [{ type: "text", text: "look" }, { type: "image", data: "QUJD", mimeType: "image/png" }], timestamp: 1 }]);
		assert.deepEqual(parts, [
			{ type: "text", text: "look" },
			{ type: "image", base64Data: "QUJD", mediaType: "image/png" },
		]);
	});

	it("renders foreign history with tool calls and results", () => {
		const text = historyText([
			user("q"),
			{ role: "assistant", content: [{ type: "toolCall", id: "t1", name: "read", arguments: { path: "a" } }], api: "x", provider: "x", model: "x", usage: {} as any, stopReason: "toolUse", timestamp: 1 } as any,
			{ role: "toolResult", toolCallId: "t1", toolName: "read", content: [{ type: "text", text: "DATA" }], isError: true, timestamp: 1 },
		]);
		assert.match(text, /<user>\nq\n<\/user>/);
		assert.match(text, /<tool_call tool="read" id="t1">\{"path":"a"\}<\/tool_call>/);
		assert.match(text, /<tool_result tool="read" id="t1" error="true">\nDATA/);
	});
});

describe("models", () => {
	it("maps pi thinking levels onto Muse efforts", () => {
		assert.equal(museEffort(undefined), "none");
		assert.equal(museEffort("xhigh"), "xhigh");
		const wonder = museModels().find((m) => m.id === "muse-wonder-1.0")!;
		assert.equal((wonder.thinkingLevelMap as any).off, null, "Muse models cannot turn reasoning off");
		assert.equal((wonder.thinkingLevelMap as any).max, "max");
		assert.equal(museEffort("off", wonder.thinkingLevelMap as any), "minimal", "no thinking = the lowest effort the model offers");
		assert.equal(museEffort(undefined, wonder.thinkingLevelMap as any), "minimal");
		const spark12 = museModels().find((m) => m.id === "muse-spark-1.2")!;
		assert.equal((spark12.thinkingLevelMap as any).max, null);
		assert.equal(wonder.cost.input, 0);
	});

	it("plans below Muse Code's own context limit", () => {
		const [m] = museModels([{ modelId: "x", contextLimit: 1_007_997, outputLimit: 64_000 }]);
		assert.ok(m.contextWindow < 1_007_997 * 0.99);
		assert.equal(m.maxTokens, 64_000);
	});
});

describe("config", () => {
	const dirs = () => {
		const root = mkdtempSync(join(tmpdir(), "pi-muse-cfg-"));
		mkdirSync(join(root, "agent"));
		mkdirSync(join(root, "proj", ".pi"), { recursive: true });
		mkdirSync(join(root, "home", ".config", "muse"), { recursive: true });
		return root;
	};

	it("lets a project tune only harmless settings", () => {
		const root = dirs();
		writeFileSync(join(root, "proj", ".pi", "muse-sdk.json"), JSON.stringify({ museBin: "/tmp/evil", contributorMode: "on", maxSessions: 2 }));
		const c = resolveConfig({ cwd: join(root, "proj"), agentDir: join(root, "agent"), home: join(root, "home"), env: {} });
		assert.equal(c.museBin, undefined);
		assert.equal(c.contributorMode, "off");
		assert.equal(c.maxSessions, 2);
	});

	it("follows the saved Muse Code contributor setting, else off", () => {
		const root = dirs();
		const where = { cwd: join(root, "proj"), agentDir: join(root, "agent"), home: join(root, "home"), env: {} as Record<string, string> };
		assert.equal(resolveConfig(where).contributorMode, "off");
		writeFileSync(join(root, "home", ".config", "muse", "settings.json"), JSON.stringify({ contributor_mode_enabled: true }));
		assert.equal(resolveConfig(where).contributorMode, "on");
		assert.equal(resolveConfig({ ...where, env: { PI_MUSE_SDK_CONTRIBUTOR: "off" } }).contributorMode, "off");
	});
});

describe("host settings", () => {
	it("never passes an API key that would switch to pay-as-you-go billing", () => {
		const env = cleanEnv({ META_API_KEY: "sk", PATH: "/bin", X: undefined });
		assert.equal(env.META_API_KEY, undefined);
		assert.equal(env.PATH, "/bin");
		assert.equal("X" in env, false);
	});

	it("runs pi's prompt and tools with Muse Code's own context and observers off", () => {
		const s = hostSettings({ systemPrompt: "P", toolset: ["mcp__pi__*"], mcpUrl: "http://127.0.0.1:1/mcp/x", skillsOff: { bundled: { "a": "off" } } }) as any;
		assert.equal(s.run.system_prompt, "P");
		assert.deepEqual(s.run.toolset, ["mcp__pi__*"]);
		assert.ok(Object.values(s.runtime_capabilities).every((v: any) => v.enabled === false));
		assert.equal(s.mcp_servers.pi.headers.Authorization, "${PI_MUSE_MCP_AUTH}", "the token stays out of the file");
		assert.ok(s.mcp_servers.pi.tool_timeout_sec >= 24 * 3600);
		assert.equal(hostSettings({ systemPrompt: "P", toolset: [], skillsOff: {} }).mcp_servers, undefined);
	});
});

let fakeSeq = 0;

/** A Muse host that answers every turn with "ok"; `close()` is what shutdown must or must not reach. */
class FakeSession implements MuseSession {
	readonly spec: HostSpec;
	readonly sessionId: string;
	model: string;
	chain = "";
	count = 0;
	awaiting = new Set<string>();
	announced = new Set<string>();
	turnId?: string;
	dead = false;
	busy = false;
	lastUsed = Date.now();
	readonly events = new AsyncQueue<HostEvent>();
	readonly exited: Promise<void>;
	readonly stderr = "";
	closed = false;
	/** When true, a turn stays in flight after its first delta until `finishTurn()`. */
	holdTurn = false;
	/** When true, a turn answers with one tool call, parking the host on pi's result. */
	toolTurn = false;
	startedTurns = 0;
	private held = false;
	private markExited: () => void = () => {};
	private turn = 0;

	constructor(spec: HostSpec) {
		this.spec = spec;
		this.model = spec.model;
		this.sessionId = `fake-${++fakeSeq}`;
		this.exited = new Promise<void>((resolve) => {
			this.markExited = resolve;
		});
	}

	async startTurn(): Promise<void> {
		this.turnId = `turn-${++this.turn}`;
		this.startedTurns += 1;
		if (this.toolTurn) {
			this.events.push({ type: "tool_calls", calls: [{ callId: "call-1", name: "mcp__pi__read", args: JSON.stringify({ path: "x" }) }] });
			return;
		}
		this.events.push({ type: "text_start", itemId: "i1" });
		this.events.push({ type: "text_delta", itemId: "i1", delta: "ok" });
		if (this.holdTurn) {
			this.held = true;
			return;
		}
		this.events.push({ type: "text_end", itemId: "i1", text: "ok" });
		this.events.push({ type: "turn_completed", turnId: this.turnId, terminal: "completed" });
	}

	/** Complete a turn held open by `holdTurn`, as if Muse Code finished responding. */
	finishTurn(): void {
		if (!this.held) return;
		this.held = false;
		this.events.push({ type: "text_end", itemId: "i1", text: "ok" });
		this.events.push({ type: "turn_completed", turnId: this.turnId ?? "", terminal: "completed" });
	}

	async steer(): Promise<void> {}
	async setModel(model: string): Promise<void> {
		this.model = model;
	}
	async interrupt(): Promise<void> {}
	deliver(): void {}

	close(): void {
		if (this.closed) return;
		this.closed = true;
		this.dead = true;
		this.events.close();
		this.markExited();
	}
}

const fakeModel: Model<Api> = {
	id: "muse-spark-1.3",
	name: "fake",
	api: API,
	provider: "muse-sdk",
	baseUrl: "",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100_000,
	maxTokens: 1000,
};
const fakeTools = [{ name: "read", description: "Read a file", parameters: Type.Object({ path: Type.String() }) }];

function fakeResources() {
	const dir = mkdtempSync(join(tmpdir(), "pi-muse-life-"));
	const bridge = new McpBridge();
	const hostEnv: HostEnv = {
		museBin: "muse",
		contributorMode: "off",
		bridge,
		baseDir: dir,
		userConfigDir: dir,
		skillsOff: async () => ({}),
		env: {},
	};
	const opened: FakeSession[] = [];
	const provider = new MuseSdkProvider({
		registry: new AnchorRegistry(join(dir, "anchors.jsonl")),
		openSession: async (spec) => {
			const s = new FakeSession(spec);
			opened.push(s);
			return s;
		},
		cwd: () => dir,
	});
	const call = async (messages: Message[]) => {
		const msg = await provider.streamSimple(fakeModel, normalizeContext({ systemPrompt: "S", tools: fakeTools, messages }), {}).result();
		return { msg, route: provider.lastRoute };
	};
	return { provider, bridge, hostEnv, opened, call };
}

describe("shared provider lifecycle", () => {
	// In pi-bolt a child (subagent) session runs in this same process and its session_shutdown fires
	// on release. That must never tear down the hosts the main session is still using: one join per
	// live session, teardown only when the last one leaves (or the process exits).
	const harness = () => {
		const r = fakeResources();
		let teardowns = 0;
		const lifecycle = new SharedLifecycle(() => {
			teardowns += 1;
			r.provider.shutdown();
			r.bridge.close();
		});
		return { ...r, lifecycle, teardowns: () => teardowns };
	};

	it("one of two sessions shutting down leaves the other's hosts open", async () => {
		const h = harness();
		const mainLeave = h.lifecycle.join();
		const childLeave = h.lifecycle.join();
		const first = await h.call([user("hi")]);
		assert.equal(first.msg.stopReason, "stop", first.msg.errorMessage ?? "");
		assert.equal(h.opened.length, 1);

		childLeave();
		assert.equal(h.lifecycle.liveSessions, 1);
		assert.equal(h.teardowns(), 0);
		assert.equal(h.opened[0].closed, false, "the main session's host must survive the child's shutdown");

		const second = await h.call([user("hi"), first.msg, user("again")]);
		assert.equal(second.route, "live", "the pooled host is still there to continue the conversation");
		assert.equal(h.opened.length, 1, "no replacement host had to be opened");
		mainLeave();
	});

	it("the last session shutting down drops every host", async () => {
		const h = harness();
		const aLeave = h.lifecycle.join();
		const bLeave = h.lifecycle.join();
		await h.call([user("hi")]);
		aLeave();
		assert.equal(h.opened[0].closed, false);
		bLeave();
		assert.equal(h.lifecycle.liveSessions, 0);
		assert.equal(h.teardowns(), 1);
		assert.equal(h.opened[0].closed, true, "with no live session left every host is dropped");
	});

	it("process exit drops every host even while sessions are live", async () => {
		const h = harness();
		const aLeave = h.lifecycle.join();
		const bLeave = h.lifecycle.join();
		await h.call([user("hi")]);
		// shutdownAll is the process "exit" handler installed for the shared lifecycle.
		h.lifecycle.shutdownAll();
		assert.equal(h.opened[0].closed, true);
		assert.equal(h.lifecycle.liveSessions, 0);
		assert.equal(h.teardowns(), 1);
		// session_shutdown handlers racing the exit must not tear down a second time.
		aLeave();
		bLeave();
		assert.equal(h.teardowns(), 1);
		assert.equal(h.lifecycle.liveSessions, 0);
	});

	it("the refcount never goes negative on double or unmatched shutdown", () => {
		const h = harness();
		const leave = h.lifecycle.join();
		leave();
		leave();
		assert.equal(h.lifecycle.liveSessions, 0);
		assert.equal(h.teardowns(), 1, "a double shutdown tears down once, not twice");
		// Leaves that arrive after a process-exit shutdown (no matching live count) are dropped.
		const aLeave = h.lifecycle.join();
		const bLeave = h.lifecycle.join();
		h.lifecycle.shutdownAll();
		aLeave();
		bLeave();
		aLeave();
		assert.equal(h.lifecycle.liveSessions, 0);
		assert.equal(h.teardowns(), 2);
	});

	it("a session replaced by /new, /resume or reload can join again after teardown", async () => {
		const h = harness();
		const leave = h.lifecycle.join();
		await h.call([user("hi")]);
		leave();
		assert.equal(h.opened[0].closed, true);
		// The replacement runtime's session_start joins again and the shared provider is reusable.
		const leaveAgain = h.lifecycle.join();
		const again = await h.call([user("fresh")]);
		assert.equal(again.msg.stopReason, "stop", again.msg.errorMessage ?? "");
		assert.equal(h.opened.length, 2, "a fresh host opens for the replaced session");
		leaveAgain();
		assert.equal(h.opened[1].closed, true);
		assert.equal(h.lifecycle.liveSessions, 0);
	});

	it("every factory in the process shares one provider, bridge and lifecycle", () => {
		let creates = 0;
		const make = () => {
			creates += 1;
			const r = fakeResources();
			return { provider: r.provider, bridge: r.bridge, hostEnv: r.hostEnv };
		};
		const exitsBefore = process.listenerCount("exit");
		const a = sharedMuse(make);
		assert.equal(process.listenerCount("exit"), exitsBefore + 1, "the first factory hooks process exit once");
		const b = sharedMuse(make);
		assert.equal(process.listenerCount("exit"), exitsBefore + 1, "later factories add no further exit hooks");
		assert.equal(creates, 1, "the second factory must not build its own provider or bridge");
		assert.equal(a.resources.provider, b.resources.provider);
		assert.equal(a.resources.bridge, b.resources.bridge);
		assert.equal(a.resources.provider.streamSimple, b.resources.provider.streamSimple);
		assert.equal(a.lifecycle, b.lifecycle);
		// Retire this generation so the factory tests below build their own.
		a.lifecycle.shutdownAll();
	});
});

/** The slice of ExtensionAPI the factory uses, capturing its handlers and registrations. */
class FakePi {
	readonly handlers = new Map<string, ((event: unknown, ctx: unknown) => unknown)[]>();
	readonly providers: { name: string; config: ProviderConfig }[] = [];

	on(event: string, handler: (event: unknown, ctx: unknown) => unknown): () => void {
		const list = this.handlers.get(event) ?? [];
		list.push(handler);
		this.handlers.set(event, list);
		return () => {};
	}

	registerProvider(name: string, config: ProviderConfig): void {
		this.providers.push({ name, config });
	}

	async emit(event: string, payload: unknown): Promise<void> {
		for (const handler of this.handlers.get(event) ?? []) await handler(payload, {});
	}

	start(): Promise<void> {
		return this.emit("session_start", { type: "session_start", reason: "startup" });
	}

	shutdown(): Promise<void> {
		return this.emit("session_shutdown", { type: "session_shutdown", reason: "quit" });
	}
}

/** Drives the real extension factory (src/index.ts) with a fake pi and fake Muse hosts. */
function factoryHarness() {
	const opened: FakeSession[] = [];
	const hostEnvs: HostEnv[] = [];
	const bridgeCloses = new Map<HostEnv, number>();
	let holdTurns = false;
	let toolTurns = false;
	const deps: MuseExtensionDeps = {
		openSession: async (spec, hostEnv) => {
			if (!hostEnvs.includes(hostEnv)) {
				hostEnvs.push(hostEnv);
				const bridge = hostEnv.bridge;
				const close = bridge.close.bind(bridge);
				bridge.close = () => {
					bridgeCloses.set(hostEnv, (bridgeCloses.get(hostEnv) ?? 0) + 1);
					close();
				};
			}
			const session = new FakeSession(spec);
			session.holdTurn = holdTurns;
			session.toolTurn = toolTurns;
			opened.push(session);
			return session;
		},
		listModels: async () => undefined,
	};
	const makePi = (): FakePi => {
		const pi = new FakePi();
		createMuseExtension(pi as unknown as ExtensionAPI, deps);
		return pi;
	};
	const streamOf = (pi: FakePi) => {
		const stream = pi.providers[pi.providers.length - 1]?.config.streamSimple;
		if (!stream) assert.fail("the factory did not register a streamSimple");
		return stream;
	};
	const call = (pi: FakePi, messages: Message[], systemPrompt = "S") => streamOf(pi)(fakeModel, normalizeContext({ systemPrompt, tools: fakeTools, messages }), {}).result();
	const waitFor = async (cond: () => boolean, what: string): Promise<void> => {
		for (let i = 0; i < 400; i++) {
			if (cond()) return;
			await new Promise((r) => setTimeout(r, 5));
		}
		assert.fail(`timed out waiting for ${what}`);
	};
	const closes = (env: HostEnv = hostEnvs[hostEnvs.length - 1]): number => bridgeCloses.get(env) ?? 0;
	return {
		opened,
		hostEnvs,
		makePi,
		call,
		streamOf,
		waitFor,
		closes,
		setHoldTurns: (v: boolean) => {
			holdTurns = v;
		},
		setToolTurns: (v: boolean) => {
			toolTurns = v;
		},
	};
}

/** The shared process-exit handler lifecycle.ts registered, found among node's exit listeners. */
function sharedExitHandler(): () => void {
	for (const listener of process.listeners("exit")) {
		const original: unknown = (listener as unknown as { listener?: unknown }).listener;
		const candidate: unknown = typeof original === "function" ? original : listener;
		if (typeof candidate === "function" && candidate.name === "piMuseSharedExit") return candidate as unknown as () => void;
	}
	assert.fail("the shared process-exit handler was not registered");
}

describe("extension factory lifecycle", () => {
	// Every test here goes through the real entry point: the default-exported factory of
	// src/index.ts, its captured pi.on handlers and the streamSimple it registers. A fake
	// openSession stands in for `muse serve`; config comes from a throwaway agent dir.
	const savedEnv = {
		agentDir: process.env.PI_CODING_AGENT_DIR,
		xdg: process.env.XDG_CONFIG_HOME,
		contributor: process.env.PI_MUSE_SDK_CONTRIBUTOR,
		testSettings: process.env.PI_MUSE_SDK_TEST_SETTINGS,
	};
	const restore = (key: string, value: string | undefined) => {
		if (value === undefined) delete process.env[key];
		else process.env[key] = value;
	};
	before(() => {
		const root = mkdtempSync(join(tmpdir(), "pi-muse-factory-"));
		process.env.PI_CODING_AGENT_DIR = join(root, "agent");
		process.env.XDG_CONFIG_HOME = join(root, "xdg");
		delete process.env.PI_MUSE_SDK_CONTRIBUTOR;
		delete process.env.PI_MUSE_SDK_TEST_SETTINGS;
	});
	after(() => {
		restore("PI_CODING_AGENT_DIR", savedEnv.agentDir);
		restore("XDG_CONFIG_HOME", savedEnv.xdg);
		restore("PI_MUSE_SDK_CONTRIBUTOR", savedEnv.contributor);
		restore("PI_MUSE_SDK_TEST_SETTINGS", savedEnv.testSettings);
	});

	it("(a) a child session shutting down leaves the main session's in-flight call alone", async () => {
		const h = factoryHarness();
		const main = h.makePi();
		await main.start();
		h.setHoldTurns(true);
		const pending = h.call(main, [user("hi")]);
		await h.waitFor(() => h.opened.length === 1 && h.opened[0].startedTurns === 1, "the main call to be in flight");

		const child = h.makePi();
		assert.equal(h.streamOf(child), h.streamOf(main), "concurrent runtimes register the one shared provider");
		await child.start();
		await child.shutdown();
		assert.equal(h.opened[0].closed, false, "the child's shutdown must not close the main session's host");

		h.opened[0].finishTurn();
		const msg = await pending;
		assert.equal(msg.stopReason, "stop", msg.errorMessage ?? "");
		assert.equal(h.opened[0].closed, false, "the host stays pooled while the main session is live");

		h.opened[0].holdTurn = false;
		const second = await h.call(main, [user("hi"), msg, user("again")]);
		assert.equal(second.stopReason, "stop", second.errorMessage ?? "");
		assert.equal(h.opened.length, 1, "the pooled host continues the conversation; none had to reopen");
		await main.shutdown();
		assert.equal(h.opened[0].closed, true, "the host drops once the main session shuts down too");
	});

	it("(b) a runtime that never emitted session_start still holds and releases a share", async () => {
		const h = factoryHarness();
		const main = h.makePi(); // no session_start on this runtime, ever
		h.setHoldTurns(true);
		const pending = h.call(main, [user("hi")]);
		await h.waitFor(() => h.opened.length === 1 && h.opened[0].startedTurns === 1, "the unstarted call to be in flight");

		const other = h.makePi();
		await other.start();
		await other.shutdown();
		assert.equal(h.opened[0].closed, false, "a call from a never-started runtime must survive another runtime's shutdown");

		h.opened[0].finishTurn();
		const msg = await pending;
		assert.equal(msg.stopReason, "stop", msg.errorMessage ?? "");

		await main.shutdown(); // never started, but its shutdown must still release its share
		assert.equal(h.opened[0].closed, true, "the unstarted runtime's shutdown releases the last share");
		assert.equal(h.closes(), 1, "the bridge closes once the last share is released");
	});

	it("(c) when all sessions shut down, idle hosts drop now and active ones after their call", async () => {
		const h = factoryHarness();
		const a = h.makePi();
		const b = h.makePi();
		await a.start();
		await b.start();
		h.setToolTurns(true);
		const first = await h.call(a, [user("hi")]);
		assert.equal(first.stopReason, "toolUse", first.errorMessage ?? "");
		assert.equal(h.opened.length, 1);
		assert.equal(h.opened[0].awaiting.size, 1, "the first host is parked on a tool call between steps");
		h.setToolTurns(false);

		h.setHoldTurns(true);
		const pending = h.call(a, [user("other")], "T"); // a different conversation: a second host
		await h.waitFor(() => h.opened.length === 2 && h.opened[1].startedTurns === 1, "the second call to be in flight");

		await b.shutdown();
		assert.equal(h.opened[0].closed, false, "nothing retires while a runtime still holds a share");
		await a.shutdown();
		assert.equal(h.opened[0].closed, true, "idle hosts, parked ones included, drop as soon as the last runtime leaves");
		assert.equal(h.opened[1].closed, false, "a host with a call in flight is never killed by teardown");
		assert.equal(h.closes(h.hostEnvs[0]), 0, "the bridge stays open until the in-flight call finishes");

		// A factory load during the drain already builds the next generation.
		const c = h.makePi();
		await c.start();
		assert.notEqual(h.streamOf(c), h.streamOf(a), "a draining generation is not handed out to a new factory");
		h.setHoldTurns(false);
		const fresh = await h.call(c, [user("hi")]);
		assert.equal(fresh.stopReason, "stop", fresh.errorMessage ?? "");
		assert.equal(h.opened.length, 3);
		assert.equal(h.opened[1].closed, false, "the draining call is still in flight on the old generation");

		h.opened[1].finishTurn();
		const msg = await pending;
		assert.equal(msg.stopReason, "stop", msg.errorMessage ?? "");
		await h.waitFor(() => h.opened[1].closed && h.closes(h.hostEnvs[0]) === 1, "the active host to drop and the old bridge to close after its call");
		assert.equal(h.opened[2].closed, false, "the old generation finishing its drain leaves the new one alone");
		await c.shutdown();
		assert.equal(h.opened[2].closed, true);
		assert.equal(h.closes(h.hostEnvs[1]), 1);
	});

	it("(d) after full retirement a new factory builds fresh resources from its own config", async () => {
		const h = factoryHarness();
		process.env.PI_MUSE_SDK_CONTRIBUTOR = "on";
		const a = h.makePi();
		await a.start();
		const first = await h.call(a, [user("hi")]);
		assert.equal(first.stopReason, "stop", first.errorMessage ?? "");
		assert.equal(h.hostEnvs[0].contributorMode, "on");
		const firstStream = h.streamOf(a);
		await a.shutdown();
		assert.equal(h.opened[0].closed, true);
		assert.equal(h.closes(h.hostEnvs[0]), 1);

		process.env.PI_MUSE_SDK_CONTRIBUTOR = "off";
		const b = h.makePi();
		await b.start();
		assert.notEqual(h.streamOf(b), firstStream, "a retired generation's provider must not be handed out again");
		const second = await h.call(b, [user("hi")]);
		assert.equal(second.stopReason, "stop", second.errorMessage ?? "");
		assert.equal(h.hostEnvs.length, 2, "the new factory must build a fresh host env after retirement");
		assert.equal(h.hostEnvs[1].contributorMode, "off", "the new generation uses the new factory's own config");
		assert.notEqual(h.hostEnvs[1].bridge, h.hostEnvs[0].bridge, "the new generation gets its own bridge");
		await b.shutdown();
		delete process.env.PI_MUSE_SDK_CONTRIBUTOR;
	});

	it("(e) double and unmatched shutdowns neither go negative nor tear down twice", async () => {
		const h = factoryHarness();
		const a = h.makePi();
		await a.start();
		const first = await h.call(a, [user("hi")]);
		assert.equal(first.stopReason, "stop", first.errorMessage ?? "");
		await a.shutdown();
		await a.shutdown();
		assert.equal(h.opened[0].closed, true);
		assert.equal(h.closes(h.hostEnvs[0]), 1, "a double shutdown tears its generation down once");

		const b = h.makePi(); // never starts
		await b.shutdown();
		await b.shutdown();
		const c = h.makePi();
		assert.notEqual(h.streamOf(c), h.streamOf(b), "the unstarted runtime's shutdown retired its generation");
		await c.start();
		const second = await h.call(c, [user("hi")]);
		assert.equal(second.stopReason, "stop", second.errorMessage ?? "");
		assert.equal(h.opened.length, 2);
		await a.shutdown(); // a stale runtime's late shutdown must not reach the live generation
		assert.equal(h.opened[1].closed, false, "a stale double shutdown cannot tear down a later generation");
		await c.shutdown();
		await c.shutdown();
		assert.equal(h.opened[1].closed, true, "the live runtime's own shutdown still retires its generation");
		assert.equal(h.closes(), 1, "that generation is torn down exactly once");
	});

	it("(f) process exit tears everything down once, even with sessions still live", async () => {
		const h = factoryHarness();
		const a = h.makePi();
		const b = h.makePi();
		await a.start();
		await b.start();
		const first = await h.call(a, [user("hi")]);
		assert.equal(first.stopReason, "stop", first.errorMessage ?? "");
		assert.equal(h.opened[0].closed, false);

		const onExit = sharedExitHandler();
		onExit();
		assert.equal(h.opened[0].closed, true, "process exit drops every host even while runtimes hold shares");
		assert.equal(h.closes(), 1);
		onExit();
		await a.shutdown(); // session_shutdown handlers racing the exit
		await b.shutdown();
		assert.equal(h.closes(), 1, "exit plus racing shutdowns tears down once, not once per handler");

		const c = h.makePi();
		await c.start();
		const second = await h.call(c, [user("hi")]);
		assert.equal(second.stopReason, "stop", second.errorMessage ?? "");
		await c.shutdown();
		assert.equal(h.opened[1].closed, true);
	});
});

describe("anchors", () => {
	it("tracks each session's last clean turn across processes", () => {
		const file = join(mkdtempSync(join(tmpdir(), "pi-muse-anchors-")), "a.jsonl");
		const r = new AnchorRegistry(file);
		r.put("k1", { sessionId: "s", turnId: "t1", cwd: "/w" });
		assert.equal(r.tail("s"), "t1");
		r.openTurn("s");
		assert.equal(r.tail("s"), undefined, "a started turn may be partial");
		const again = new AnchorRegistry(file);
		assert.deepEqual(again.get("k1"), { sessionId: "s", turnId: "t1", cwd: "/w" });
		assert.equal(again.tail("s"), undefined);
	});
});
