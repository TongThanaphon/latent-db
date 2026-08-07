//! Debug-only allocation tracking, for `*_alloc_check` tests that assert a
//! hot path performs zero (or a known number of) heap allocations.
//!
//! Shape follows katgpt-rs's `katgpt-core::alloc` module: counters are
//! **per-thread** (thread-local `Cell`, not a process-global atomic) so
//! parallel tests don't bleed each other's allocations into their counts,
//! following a `reset → measure-on-calling-thread → get` protocol.
//!
//! **Whole-module `debug_assertions` gate:** the `#![cfg(debug_assertions)]`
//! inner attribute below gates every item in this file at once -- in a
//! release build the module compiles to nothing (see `lib.rs`'s
//! `#[global_allocator]` install, which is gated the same way), leaving the
//! plain `System` allocator with no per-allocation counting overhead.

#![cfg(debug_assertions)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Aggregate per-thread allocation stats (count + bytes). `Copy` so it can
/// live in a `Cell` (single load + store per `alloc`, no `RefCell` overhead).
#[derive(Clone, Copy)]
struct AllocStats {
    count: usize,
    bytes: usize,
}

impl AllocStats {
    /// `const` constructor so the `thread_local!` initializer is const-evaluable
    /// -- non-const init would itself allocate on first touch, which inside
    /// `alloc()` would recurse without bound.
    const ZERO: Self = Self { count: 0, bytes: 0 };
}

thread_local! {
    static THREAD_ALLOC: Cell<AllocStats> = const { Cell::new(AllocStats::ZERO) };
}

/// Debug-only allocator wrapper that tracks allocation count and bytes on the
/// **calling thread**. Installed as `#[global_allocator]` in `lib.rs`.
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        THREAD_ALLOC.with(|cell| {
            let mut s = cell.get();
            s.count = s.count.wrapping_add(1);
            s.bytes = s.bytes.wrapping_add(layout.size());
            cell.set(s);
        });
        // Safety: delegated to the system allocator, layout is valid.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Safety: delegated to the system allocator, ptr+layout are valid.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Reset the **calling thread's** allocation counters to zero.
pub fn reset_alloc_stats() {
    THREAD_ALLOC.with(|cell| cell.set(AllocStats::ZERO));
}

/// Get the **calling thread's** `(allocation_count, total_bytes)` since the
/// last [`reset_alloc_stats`] on this thread.
pub fn get_alloc_stats() -> (usize, usize) {
    THREAD_ALLOC.with(|cell| {
        let s = cell.get();
        (s.count, s.bytes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_clears_stats() {
        reset_alloc_stats();
        let (count, bytes) = get_alloc_stats();
        assert!(
            count <= 5,
            "count should be near-zero after reset, got {count}"
        );
        assert!(
            bytes <= 4096,
            "bytes should be near-zero after reset, got {bytes}"
        );
    }

    #[test]
    fn test_alloc_increments_count() {
        reset_alloc_stats();
        let _v: Vec<u8> = vec![0u8; 1024];
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "at least one allocation should have occurred");
        assert!(bytes >= 1024, "bytes should be at least 1024, got {bytes}");
    }

    #[test]
    fn test_multiple_allocs_accumulate() {
        reset_alloc_stats();
        let _v1: Vec<u8> = vec![0u8; 64];
        let _v2: Vec<u8> = vec![0u8; 128];
        let (count, bytes) = get_alloc_stats();
        assert!(count >= 2, "at least two allocations, got {count}");
        assert!(bytes >= 192, "bytes should be at least 192, got {bytes}");
    }

    /// Thread isolation: allocations on another thread are not visible on
    /// this thread. `*_alloc_check` tests rely on this to measure a single
    /// call path without cross-thread noise.
    #[test]
    fn test_thread_isolation() {
        const WORKER_ALLOC_BYTES: usize = 4096;
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            reset_alloc_stats();
            let _v: Vec<u8> = vec![0u8; WORKER_ALLOC_BYTES];
            let (_count, bytes) = get_alloc_stats();
            let _ = tx.send(bytes);
        });
        // Reset on THIS thread AFTER spawning -- thread spawn itself
        // allocates bookkeeping on the spawning thread, which is legitimate
        // runtime overhead, not the property under test.
        reset_alloc_stats();
        let worker_bytes = rx.recv().expect("worker should report");
        handle.join().expect("worker thread panicked");
        let (_main_count, main_bytes) = get_alloc_stats();
        assert!(
            worker_bytes >= WORKER_ALLOC_BYTES,
            "worker thread should have seen its own {WORKER_ALLOC_BYTES}-byte \
             allocation, got {worker_bytes}"
        );
        assert!(
            main_bytes < WORKER_ALLOC_BYTES,
            "main thread should not see the worker's {WORKER_ALLOC_BYTES}-byte \
             allocation, but observed {main_bytes} bytes -- thread-local \
             isolation is broken"
        );
    }
}
