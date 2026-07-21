/* tslint:disable */
/* eslint-disable */

export class WasmLatentDb {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    dim(): number;
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
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmlatentdb_free: (a: number, b: number) => void;
    readonly wasmlatentdb_dim: (a: number) => number;
    readonly wasmlatentdb_fromBytes: (a: number, b: number) => [number, number, number];
    readonly wasmlatentdb_isEmpty: (a: number) => number;
    readonly wasmlatentdb_len: (a: number) => number;
    readonly wasmlatentdb_search: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
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
