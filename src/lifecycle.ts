import type { McpBridge } from "./bridge.ts";
import type { HostEnv } from "./host.ts";
import type { MuseSdkProvider } from "./provider.ts";

/** The process-wide parts every pi session in this process must share. */
export interface SharedMuse {
	provider: MuseSdkProvider;
	bridge: McpBridge;
	hostEnv: HostEnv;
}

export interface SharedHandle {
	resources: SharedMuse;
	lifecycle: SharedLifecycle;
}

/** Why a shared generation is being torn down. */
export type TeardownReason = "last-leave" | "shutdown-all";

/**
 * Refcounted teardown for resources shared by every pi session in one process.
 *
 * pi-bolt subagents run as extra pi sessions in the main session's process. Each session
 * loads this extension (a fresh factory call), all sessions share one model registry (so
 * the last `registerProvider` wins for everyone), and a released child session emits
 * `session_shutdown` just like a quitting main session. Tearing the provider and bridge
 * down on any single `session_shutdown` therefore kills hosts another live session is
 * using mid-turn. Instead each extension runtime holds one count from the moment its
 * factory runs (`join()` in the factory, not on `session_start`: a registered
 * `streamSimple` can be called before, or entirely without, its runtime ever emitting
 * `session_start`) and the teardown runs only when the last runtime leaves on
 * `session_shutdown`, or unconditionally on process exit.
 */
export class SharedLifecycle {
	private count = 0;
	private tornDown = false;
	private readonly teardown: (reason: TeardownReason) => void;

	constructor(teardown: (reason: TeardownReason) => void) {
		this.teardown = teardown;
	}

	/** Sessions currently holding the shared resources. Never negative. */
	get liveSessions(): number {
		return this.count;
	}

	/**
	 * One runtime started using the shared resources. The returned leave function ends that
	 * one hold and is idempotent: calling it twice (a double `session_shutdown`) or after
	 * `shutdownAll()` (a shutdown racing process exit) neither drives the count negative
	 * nor tears down a second time. A replacement runtime (`/new`, `/resume`, reload)
	 * simply joins again after its predecessor left.
	 */
	join(): () => void {
		this.count += 1;
		this.tornDown = false;
		let held = true;
		return () => {
			if (!held) return;
			held = false;
			if (this.count === 0) return; // shutdownAll() already ran; nothing left to release
			this.count -= 1;
			if (this.count === 0 && !this.tornDown) {
				this.tornDown = true;
				this.teardown("last-leave");
			}
		};
	}

	/** Process exit: tear down regardless of how many runtimes still hold a count, once. */
	shutdownAll(): void {
		this.count = 0;
		if (this.tornDown) return;
		this.tornDown = true;
		this.teardown("shutdown-all");
	}
}

interface SharedState {
	/** The generation new factory calls join; cleared the moment a generation retires. */
	handle?: SharedHandle;
	/** Every generation whose bridge is not closed yet, including one still draining. */
	live: Set<SharedMuse>;
}

// Symbol.for + globalThis, not a module-level variable: pi reloads extensions (session
// replacement, /reload, each child session) and the reloaded module must find the same
// provider, bridge and refcount instead of building a second set.
const SHARED_KEY: unique symbol = Symbol.for("pi-muse-subscription:shared-v1");
type SharedGlobal = typeof globalThis & { [SHARED_KEY]?: SharedState };

function sharedState(): SharedState {
	const g = globalThis as SharedGlobal;
	let state = g[SHARED_KEY];
	if (!state) {
		const created: SharedState = { live: new Set() };
		g[SHARED_KEY] = created;
		// Registered once for the process, not once per factory call: every session in the
		// process loads this extension, and their combined hosts must all go at exit. The
		// current generation goes through its lifecycle; a retired generation that is still
		// draining an active call is no longer reachable that way, so it is shut hard here.
		process.once("exit", function piMuseSharedExit() {
			created.handle?.lifecycle.shutdownAll();
			for (const resources of [...created.live]) {
				resources.provider.shutdown();
				resources.bridge.close();
			}
			created.live.clear();
		});
		state = created;
	}
	return state;
}

/**
 * The one provider, bridge and lifecycle for this process. The first factory call of a
 * generation builds them via `create`; later calls while that generation is live (child
 * sessions, reloads) return the same handle and never invoke `create` again, so every
 * registered `streamSimple` is the same one no matter which session's registration the
 * shared model registry kept last. Once a generation retires (last runtime left, or
 * process exit) the handle is cleared before anything is torn down, so the next factory
 * call builds a fresh generation from its own config instead of keeping the retired
 * generation's (e.g. its `hostEnv.contributorMode`).
 */
export function sharedMuse(create: () => SharedMuse): SharedHandle {
	const state = sharedState();
	if (!state.handle) {
		const resources = create();
		state.live.add(resources);
		state.handle = {
			resources,
			lifecycle: new SharedLifecycle((reason) => {
				if (state.handle?.resources === resources) state.handle = undefined;
				if (reason === "shutdown-all") {
					// Process exit: nothing can finish, so drop every host at once.
					resources.provider.shutdown();
					resources.bridge.close();
					state.live.delete(resources);
				} else {
					// Last runtime left: idle hosts drop now, a host with a call still in
					// flight drops when that call finishes, and the bridge closes after it.
					resources.provider.retire(() => {
						resources.bridge.close();
						state.live.delete(resources);
					});
				}
			}),
		};
	}
	return state.handle;
}
