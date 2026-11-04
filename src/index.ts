import { appendFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { type ExtensionAPI, getAgentDir, getDocsPath, getExamplesPath, getReadmePath } from "@earendil-works/pi-coding-agent";
import { AnchorRegistry } from "./anchors.ts";
import { McpBridge } from "./bridge.ts";
import { museConfigDir, resolveConfig } from "./config.ts";
import { listSkillsOff, type SkillsOff, sweepHostConfigs } from "./home.ts";
import { type HostEnv, type HostSpec, listModels, MuseHost, type MuseSession } from "./host.ts";
import { sharedMuse } from "./lifecycle.ts";
import { type CatalogRow, museModels } from "./models.ts";
import { applyMusePrompt } from "./prompt.ts";
import { API, MuseSdkProvider } from "./provider.ts";

export const PROVIDER = "muse-sdk";

/**
 * Host seams for tests. pi loads the factory with only `pi`, so both stay at their
 * production defaults there; tests pass a fake `openSession` (and keep `listModels`
 * from spawning a real `muse serve`) to drive the real factory offline. Only the
 * factory call that builds a shared generation uses them; later factory calls in the
 * same generation share its provider, bridge and host env (see lifecycle.ts).
 */
export interface MuseExtensionDeps {
	openSession?: (spec: HostSpec, hostEnv: HostEnv) => Promise<MuseSession>;
	listModels?: (hostEnv: HostEnv) => Promise<CatalogRow[] | undefined>;
}

/**
 * pi extension: Meta's Muse models through the Muse Code SDK, authenticated by the local Muse Code
 * login (subscription). pi keeps its own agent loop, tools, prompt and compaction; a `muse serve`
 * host is used only as the authenticated model transport.
 */
export default function (pi: ExtensionAPI, deps: MuseExtensionDeps = {}) {
	const env = process.env;
	const config = resolveConfig({ cwd: process.cwd(), agentDir: getAgentDir(), home: homedir(), env });
	const baseDir = join(getAgentDir(), "muse-sdk");
	const debugPath = env.PI_MUSE_SDK_DEBUG;
	const debug = debugPath ? (msg: string) => appendFileSync(debugPath, `${new Date().toISOString()} ${msg}\n`) : undefined;
	const museBin = config.museBin ?? "muse";
	const userConfigDir = museConfigDir(env, homedir());
	sweepHostConfigs(baseDir);

	// One provider and bridge for the whole process, shared by every pi session in it
	// (the main session and pi-bolt's in-process subagents alike): all sessions share one
	// model registry, so any of them may end up serving another's turn, and one session's
	// shutdown must not close hosts another live session is using. See lifecycle.ts.
	const shared = sharedMuse(() => {
		let skills: Promise<SkillsOff> | undefined;
		const bridge = new McpBridge();
		const hostEnv: HostEnv = {
			museBin,
			contributorMode: config.contributorMode,
			bridge,
			baseDir,
			userConfigDir,
			skillsOff: () => (skills ??= listSkillsOff(museBin, baseDir, env)),
			env,
			debug,
			// Test hook: extra Muse Code settings, e.g. a fake model endpoint for offline end-to-end tests.
			...(env.PI_MUSE_SDK_TEST_SETTINGS ? { settingsOverlay: JSON.parse(env.PI_MUSE_SDK_TEST_SETTINGS) } : {}),
		};
		const provider = new MuseSdkProvider({
			registry: new AnchorRegistry(join(baseDir, "anchors.jsonl")),
			openSession: (spec) => (deps.openSession ? deps.openSession(spec, hostEnv) : MuseHost.open(spec, hostEnv)),
			...(config.maxSessions ? { maxSessions: config.maxSessions } : {}),
			...(config.idleMinutes ? { idleMs: config.idleMinutes * 60_000 } : {}),
			debug,
		});
		return { provider, bridge, hostEnv };
	});
	const { provider, hostEnv } = shared.resources;
	const lifecycle = shared.lifecycle;
	// This runtime's hold on the shared provider, taken now rather than on session_start:
	// registerProvider below makes this runtime's streamSimple reachable through the
	// process-wide model registry at once, and a call may be served by it before (or
	// without) this runtime ever emitting session_start. session_shutdown releases the
	// hold whether or not a start ever arrived; clearing the reference (and the leave
	// function itself) keeps a double shutdown from releasing twice.
	let leaveShared: (() => void) | undefined = lifecycle.join();

	const register = (models = museModels()) =>
		pi.registerProvider(PROVIDER, {
			name: "Muse (subscription via Muse Code SDK)",
			// Muse Code authenticates itself; pi only needs a non-empty key to treat the provider as configured.
			apiKey: "muse-code-login",
			api: API as any,
			baseUrl: "muse-code-sdk://local",
			models,
			streamSimple: provider.streamSimple,
		});
	register();

	// Pick up the models the user's plan offers from Muse Code's live catalog.
	let refreshed = false;
	const refreshCatalog = deps.listModels ?? listModels;
	pi.on("session_start", () => {
		if (refreshed) return;
		refreshed = true;
		void refreshCatalog(hostEnv).then((rows) => {
			if (rows) register(museModels(rows));
		});
	});

	const docs = { readme: getReadmePath(), docs: getDocsPath(), examples: getExamplesPath() };
	pi.on("before_agent_start", (event, ctx) => applyMusePrompt(event, ctx.model, PROVIDER, config.promptMode ?? "muse", docs));

	// Only the last live runtime's shutdown retires the shared generation (process exit,
	// hooked in lifecycle.ts, shuts it down at once); any other shutdown just leaves.
	// Retirement drops idle hosts now but lets a call still in flight finish first.
	pi.on("session_shutdown", () => {
		leaveShared?.();
		leaveShared = undefined;
	});
}
