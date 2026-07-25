/* @ts-self-types="./latent_db.d.ts" */

export class WasmLatentDb {
    static __wrap(ptr) {
        const obj = Object.create(WasmLatentDb.prototype);
        obj.__wbg_ptr = ptr;
        WasmLatentDbFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLatentDbFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlatentdb_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    dim() {
        const ret = wasm.wasmlatentdb_dim(this.__wbg_ptr);
        return ret >>> 0;
    }
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
     * @param {Float32Array} anchor
     * @param {number} neighborhood_size
     * @param {number} k_nearest
     * @param {number} steps
     * @param {bigint} seed
     * @returns {string}
     */
    exploreNeighborhood(anchor, neighborhood_size, k_nearest, steps, seed) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passArrayF32ToWasm0(anchor, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.wasmlatentdb_exploreNeighborhood(this.__wbg_ptr, ptr0, len0, neighborhood_size, k_nearest, steps, seed);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * Deserialize a `LatentDb` previously produced by `bincode::serialize`
     * (see `examples/build_index.rs`), e.g. from a `fetch()`'d byte buffer.
     * @param {Uint8Array} bytes
     * @returns {WasmLatentDb}
     */
    static fromBytes(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlatentdb_fromBytes(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmLatentDb.__wrap(ret[0]);
    }
    /**
     * @returns {boolean}
     */
    isEmpty() {
        const ret = wasm.wasmlatentdb_isEmpty(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    len() {
        const ret = wasm.wasmlatentdb_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Approximate nearest-neighbour search. `query` is the caller's own
     * pre-computed embedding (this crate never computes embeddings itself
     * -- see README). Returns a JSON string: `[{"id":..,"score":..,"metadata":".."}, ...]`.
     * @param {Float32Array} query
     * @param {number} k
     * @param {number} nprobe
     * @returns {string}
     */
    search(query, k, nprobe) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passArrayF32ToWasm0(query, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.wasmlatentdb_search(this.__wbg_ptr, ptr0, len0, k, nprobe);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * Like [`Self::search`], but first shifts `query` by a steering
     * direction before searching -- see `steering::SteeringVector`.
     * `direction` must be unit-norm (within `norm_tol`) and the same
     * dimension as the DB; `alpha` (steering strength) must be in `[0, 1]`.
     * Returns the same JSON shape as [`Self::search`].
     * @param {Float32Array} query
     * @param {number} k
     * @param {number} nprobe
     * @param {Float32Array} direction
     * @param {number} alpha
     * @param {number} norm_tol
     * @returns {string}
     */
    searchSteered(query, k, nprobe, direction, alpha, norm_tol) {
        let deferred4_0;
        let deferred4_1;
        try {
            const ptr0 = passArrayF32ToWasm0(query, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(direction, wasm.__wbindgen_malloc);
            const len1 = WASM_VECTOR_LEN;
            const ret = wasm.wasmlatentdb_searchSteered(this.__wbg_ptr, ptr0, len0, k, nprobe, ptr1, len1, alpha, norm_tol);
            var ptr3 = ret[0];
            var len3 = ret[1];
            if (ret[3]) {
                ptr3 = 0; len3 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred4_0 = ptr3;
            deferred4_1 = len3;
            return getStringFromWasm0(ptr3, len3);
        } finally {
            wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
        }
    }
}
if (Symbol.dispose) WasmLatentDb.prototype[Symbol.dispose] = WasmLatentDb.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./latent_db_bg.js": import0,
    };
}

const WasmLatentDbFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlatentdb_free(ptr, 1));

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('latent_db_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
