/* tslint:disable */
/* eslint-disable */

export class WasmLatentDb {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    dim(): number;
    /**
     * Explore a Viable Manifold Graph over the `neighborhood_size` records
     * closest (Euclidean, over the approximate decoded vector) to `anchor`
     * -- typically the same embedding just passed to `search`. Reports
     * graph size, a random walk of `steps` hops (seeded by `seed`) starting
     * near `anchor`, and -- when a second in-graph record is found -- a
     * geodesic between the two nearest in-graph records to `anchor`.
     *
     * The Euclidean radius that gates graph membership is derived from
     * `neighborhood_size` (the distance to the `neighborhood_size`-th
     * closest record) rather than taken as a raw parameter: a fixed radius
     * would have to be picked in the caller's embedding-distance units,
     * which vary by model and aren't knowable in advance, whereas "include
     * my N nearest records" is scale-invariant and can never silently
     * produce an empty graph the way a mis-scaled fixed radius can.
     *
     * Returns a JSON object:
     * `{ nNodes, nEdges, radius, randomWalk: [{id,metadata}, ...],
     *    geodesic: { from, to, hops, path: [{id,metadata}, ...] } | null }`.
     */
    exploreNeighborhood(anchor: Float32Array, neighborhood_size: number, k_nearest: number, steps: number, seed: bigint): string;
    /**
     * Deserialize a `LatentDb` previously produced by `bincode::serialize`
     * (see `examples/build_index.rs`), e.g. from a `fetch()`'d byte buffer.
     */
    static fromBytes(bytes: Uint8Array): WasmLatentDb;
    isEmpty(): boolean;
    len(): number;
    /**
     * Approximate nearest-neighbour search. `query` is the caller's own
     * pre-computed embedding (this crate never computes embeddings itself
     * -- see README). Returns a JSON string: `[{"id":..,"score":..,"metadata":".."}, ...]`.
     */
    search(query: Float32Array, k: number, nprobe: number): string;
    /**
     * Like [`Self::search`], but first shifts `query` by a steering
     * direction before searching -- see `steering::SteeringVector`.
     * `direction` must be unit-norm (within `norm_tol`) and the same
     * dimension as the DB; `alpha` (steering strength) must be in `[0, 1]`.
     * Returns the same JSON shape as [`Self::search`].
     */
    searchSteered(query: Float32Array, k: number, nprobe: number, direction: Float32Array, alpha: number, norm_tol: number): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmlatentdb_free: (a: number, b: number) => void;
    readonly wasmlatentdb_dim: (a: number) => number;
    readonly wasmlatentdb_exploreNeighborhood: (a: number, b: number, c: number, d: number, e: number, f: number, g: bigint) => [number, number, number, number];
    readonly wasmlatentdb_fromBytes: (a: number, b: number) => [number, number, number];
    readonly wasmlatentdb_isEmpty: (a: number) => number;
    readonly wasmlatentdb_len: (a: number) => number;
    readonly wasmlatentdb_search: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmlatentdb_searchSteered: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
