# SLM-A macOS memory guard audit — 2026-10-12

## Decision

The former 750 MB `psutil.swap_memory().free` preflight floor incorrectly treated **unused space in currently allocated swap files** as though it were a fixed macOS swap limit. The 4,096 MB `vm.swapusage.total` reading is the current allocation, not a promise of a maximum or of future capacity. Apple describes a dynamic pager that creates another backing-store file when the pool reaches a high-water mark, and current XNU source includes a swap-file creation thread with capacity and failure checks. More swap is therefore possible, but not guaranteed. Disk space, system policy, and memory pressure remain material limits. [Apple VM guide](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/vm/vm.html); [Apple XNU swap implementation](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_compressor_backing_store.c).

| Measurement | Meaning and use in the corrected guard |
| --- | --- |
| `vm.swapusage` total/used/free | Current swap-file allocation and occupancy. Logged, not used as a fixed-capacity veto. |
| `swap_memory().sout` | Cumulative bytes moved out to disk; its **difference** over the preflight window and run records new activity. A >600 MB increase aborts. [psutil API](https://psutil.io/api/). |
| Swap used growth | >600 MB from launch aborts, preserving the previous growth ceiling. |
| `memory_pressure -Q` | Logged percentage; the original 20% preflight and 10% live floors remain. Apple's broader Activity Monitor pressure combines free memory, swap rate, wired memory, and file cache, so this CLI percentage is not a complete safety proof. [Apple Activity Monitor guide](https://support.apple.com/en-lamr/guide/activity-monitor/actmntr1004/mac). |
| Available RAM and disk | Original 1.2 GB preflight/700 MB live RAM floors and 1 GB disk floor remain. Disk is checked **during** execution too. On this machine the scratch and VM APFS volumes report the same available space. |
| MLX active/peak/cache | Separate allocator metrics in the worker. Active/peak omit cached buffers; RSS and these metrics must not be added as if disjoint. [MLX active memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html); [MLX cache memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html); [MLX peak memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html). |

No numeric threshold was lowered in `config.v1.json`; its hash remains `2882cf15f702719608d597050640d6d5fd75be73570b62e4ce51b7aa99501998`. The historical 750 MB preflight and 400 MB live *allocated-swap-free* fields are now informational on macOS. The existing swap-used growth and other memory limits remain active. The parent records two preflight samples about one second apart, then samples live resources approximately every 0.5 seconds. This is a bounded risk check, not a guarantee that macOS can always create more swap.

## Verified bounded smoke test

The corrected preflight passed with 1,754,693,632 bytes available RAM, 42% reported free memory pressure, 29,856,927,744 bytes disk headroom, and no swap-out growth in its one-second sample. Already allocated swap had only 719,388,672 bytes free, so the old 750 MB rule would have refused. The new result is saved in `runs/v1-smoke-memory-audit/`.

| Metric | Measured result |
| --- | ---: |
| Status | Completed; exit 0; exactly 5 steps |
| Synthetic training tokens | 1,280 (256 per step) |
| Aggregate measured-step throughput | 6,452 tokens/s, including a 0.161 s first step |
| Per-step duration range | 0.0069–0.1610 s |
| Parent wall time | 1.022 s |
| Peak MLX active allocation | 81,512,964 bytes |
| MLX active / cache at worker completion | 21,385,260 / 70,807,100 bytes |
| Maximum sampled worker RSS | 53,297,152 bytes |
| Minimum sampled available RAM | 1,645,166,592 bytes |
| Minimum reported free memory pressure | 40% |
| Swap used growth | 0 bytes |
| Swap-out counter growth | 163,840 bytes at maximum sampled point |
| Minimum sampled disk headroom | 29,851,136,000 bytes |
| Initial / final train loss on first window | 7.668 / 7.899 |
| Initial / final validation loss | 7.740 / 7.971 |

The five per-step losses are in `steps.jsonl`; the fifth step used a different window and its loss is not directly comparable with the first. Loss did not improve on the fixed first window or validation after five steps. This run demonstrates only that the scratch implementation can complete a bounded, guarded optimization pass under these conditions. It does **not** establish useful language modelling or academic QA ability, longer-run safety, or feasibility of a larger model. No overfit test or generation was run for this audit.

All seven focused tests passed, including the new low-allocated-swap, swap growth, swap-out activity, disk, and memory-pressure assertions. The test log is `checks/v2-memory-guard-unittest.log`. The five-step checkpoint metadata reports 1,332,352 parameters and step 5; both model and AdamW file hashes were verified after the run. `reproducibility.v2.json` records the exact source, data, environment, and configuration hashes. The previous `reproducibility.v1.json` and refused-run artifacts remain unchanged.

## Recommendation

**GO for this exact five-step synthetic smoke test on the measured machine; completed without a guard stop. NO-GO for extrapolating to longer training or a larger model from this result.** Future experiments need a fresh preflight and the same live abort conditions; rising swap-out activity, pressure, or disk use must be treated as a stop signal even if `vm.swapusage.free` remains positive.
