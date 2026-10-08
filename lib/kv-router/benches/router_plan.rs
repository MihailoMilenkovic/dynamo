// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request routing cost of the Plan-returning `Router` against the direct
//! `select_and_reserve` path, over real selection cores with identical
//! workers, prompt, cache state and policies.
//!
//! Each iteration routes one request to completion and releases every
//! booking. `direct/*` is today's path (one `select_and_reserve` per stage,
//! hand-composed); `plan/*` is `Router::plan` + `schedule` over
//! `SelectionCore` and `MultiStageRouter`.
//!
//! `ROUTER_BENCH_PERCENTILES=<samples>` replaces criterion with a fixed-sample
//! run that prints p50/p95/p99 per scenario; `ROUTER_BENCH_WORKERS` sets the
//! workers per set (default 8).

use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use dynamo_kv_router::conditional_disagg::IslBoundingPolicy;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::indexer::KvIndexerInterface;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, RoutingConstraints, StorageTier,
    compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::router::{
    ClassTable, MultiStageRouter, Outcome, Router, SkipRule, StageList,
};
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, RouterConfigOverride, WorkerType};
use tokio_util::sync::CancellationToken;

const BLOCK_SIZE: u32 = 16;
const MODEL: &str = "default";

fn prompt_tokens() -> Vec<u32> {
    (1..=256).collect()
}

fn workers_per_set() -> u64 {
    std::env::var("ROUTER_BENCH_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8)
}

/// A core for one worker set with `workers` single-rank workers, ids
/// `base + 1 ..= base + workers`.
fn core_for(
    runtime: &tokio::runtime::Runtime,
    worker_type: WorkerType,
    base: u64,
    workers: u64,
) -> Arc<SelectionCore> {
    let config = KvRouterConfig {
        use_kv_events: true,
        router_queue_threshold: None,
        ..Default::default()
    };
    let core = SelectionCore::try_new_local_for(
        worker_type,
        config,
        1,
        CancellationToken::new(),
        SelectionCacheConfig::default(),
        Arc::new(|config, role, _| {
            dynamo_kv_router::WorkerSelectionPolicy::reference(
                config.clone(),
                role.default_selector_label(),
            )
        }),
    )
    .expect("core");
    runtime.block_on(async {
        for worker_id in (base + 1)..=(base + workers) {
            core.upsert_worker(WorkerRequest {
                worker_id,
                endpoint: Some(format!("http://worker-{worker_id}:8000")),
                kv_events_endpoint: Some(format!("tcp://127.0.0.1:{}", 40_000 + worker_id)),
                block_size: Some(BLOCK_SIZE),
                max_num_batched_tokens: Some(8192),
                ..WorkerRequest::default()
            })
            .await
            .expect("upsert");
        }
    });
    Arc::new(core)
}

/// Record the whole prompt as cached on `worker_id`, so a preview of this
/// core sees a full prefix hit there.
async fn seed_prefix(core: &SelectionCore, worker_id: u64) {
    let key = RoutingPartitionId::new(MODEL, "default");
    let local_hashes =
        compute_block_hash_for_seq(&prompt_tokens(), BLOCK_SIZE, BlockHashOptions::default());
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
    let blocks = local_hashes
        .iter()
        .zip(sequence_hashes.iter())
        .map(|(&tokens_hash, &sequence_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(sequence_hash),
            tokens_hash,
            mm_extra_info: None,
        })
        .collect();
    let indexer = core.partition(&key).expect("partition").indexer().clone();
    indexer
        .apply_event_routed(RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        ))
        .await
        .expect("seed index");
    if let Indexer::Single { primary, .. } = &indexer {
        primary.flush().await;
    }
}

fn request(id: &str) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: MODEL.to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(id.to_string()),
        prompt: PromptRequest {
            token_ids: Some(prompt_tokens()),
            ..PromptRequest::default()
        },
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        policy_class: None,
        all_now: false,
    }
}

/// Route one request to completion over a `Router` and release everything:
/// book what is bookable, forward what is ready, repeat.
async fn route(router: &dyn Router, req: &SelectAndReserveRequest) {
    let mut plan = router.plan(req).expect("plan");
    loop {
        router.schedule(req, &mut plan).await.expect("schedule");
        let ready: Vec<usize> = plan.ready().collect();
        if ready.is_empty() {
            break;
        }
        for k in ready {
            let attempt = plan.dispatch(k).expect("dispatch");
            let worker = plan.worker(k).expect("worker");
            plan.complete(
                k,
                attempt,
                Outcome {
                    worker,
                    kv_hint: None,
                },
            )
            .expect("complete");
        }
    }
    assert!(!plan.has_pending(), "a stage was never booked");
    plan.abort();
}

/// Today's disaggregated composition in library terms: prefill, then decode
/// with the remote-prefill override the frontend sets.
async fn direct_prefill_decode(prefill: &SelectionCore, decode: &SelectionCore, id: &str) {
    prefill
        .select_and_reserve(request(&format!("{id}/p")))
        .await
        .expect("prefill");
    let mut decode_request = request(&format!("{id}/d"));
    decode_request.router_config_override = Some(RouterConfigOverride {
        track_prefill_tokens: Some(false),
        assume_kv_reuse: Some(false),
        ..Default::default()
    });
    decode
        .select_and_reserve(decode_request)
        .await
        .expect("decode");
    prefill
        .free_reservation(&format!("{id}/p"))
        .await
        .expect("free prefill");
    decode
        .free_reservation(&format!("{id}/d"))
        .await
        .expect("free decode");
}

/// Routes request number `sequence` once, on the given runtime.
type Route = Box<dyn FnMut(&tokio::runtime::Runtime, u64)>;

/// One scenario: a name and the work of routing one request.
struct Scenario {
    name: String,
    run: Route,
}

fn scenarios(runtime: &tokio::runtime::Runtime, workers: u64) -> Vec<Scenario> {
    let aggregated = core_for(runtime, WorkerType::Aggregated, 0, workers);
    let prefill = core_for(runtime, WorkerType::Prefill, 100, workers);
    let decode = core_for(runtime, WorkerType::Decode, 200, workers);
    // A second decode set with the prompt cached on one worker: the bypass case.
    let cached_decode = core_for(runtime, WorkerType::Decode, 300, workers);
    runtime.block_on(seed_prefix(&cached_decode, 301));

    let multistage = |list: StageList, decode: &Arc<SelectionCore>, conditional: bool| {
        let mut builder = MultiStageRouter::builder()
            .set(WorkerType::Aggregated, aggregated.clone())
            .set(WorkerType::Prefill, prefill.clone())
            .set(WorkerType::Decode, decode.clone())
            .classes(ClassTable::new(list));
        if conditional {
            builder = builder
                .conditional_disagg(Arc::new(IslBoundingPolicy::new(true, 2048, 0.7)), false);
        }
        Arc::new(builder.build().expect("router"))
    };
    let one_set: Arc<dyn Router> = aggregated.clone();
    let planned = |name: &str, router: Arc<dyn Router>, all_now: bool| {
        let name = name.to_string();
        Scenario {
            name: name.clone(),
            run: Box::new(move |runtime, sequence| {
                let mut req = request(&format!("{name}-{sequence}"));
                req.all_now = all_now;
                runtime.block_on(route(router.as_ref(), &req));
            }),
        }
    };
    let mut conditional_list = StageList::prefill_decode();
    conditional_list.stages[0].skip = Some(SkipRule::ConditionalDisagg);
    // The two conditional scenarios must take different paths, or the
    // comparison measures nothing.
    for (decode, expect_skipped) in [(&decode, false), (&cached_decode, true)] {
        let router = multistage(conditional_list.clone(), decode, true);
        let req = request("probe");
        let mut plan = router.plan(&req).expect("plan");
        runtime
            .block_on(router.schedule(&req, &mut plan))
            .expect("schedule");
        let skipped = plan.state_of(0) == Some(&dynamo_kv_router::router::StageState::Skipped);
        assert_eq!(
            skipped, expect_skipped,
            "conditional probe took the wrong path"
        );
        plan.abort();
    }

    let direct_aggregated = aggregated.clone();
    let (direct_prefill, direct_decode) = (prefill.clone(), decode.clone());
    vec![
        Scenario {
            name: "direct/aggregated".to_string(),
            run: Box::new(move |runtime, sequence| {
                let id = format!("direct-{sequence}");
                runtime.block_on(async {
                    direct_aggregated
                        .select_and_reserve(request(&id))
                        .await
                        .expect("reserve");
                    direct_aggregated.free_reservation(&id).await.expect("free");
                });
            }),
        },
        planned("plan/aggregated", one_set, false),
        Scenario {
            name: "direct/prefill_decode".to_string(),
            run: Box::new(move |runtime, sequence| {
                runtime.block_on(direct_prefill_decode(
                    &direct_prefill,
                    &direct_decode,
                    &format!("direct-pd-{sequence}"),
                ));
            }),
        },
        planned(
            "plan/prefill_decode",
            multistage(StageList::prefill_decode(), &decode, false),
            false,
        ),
        planned(
            "plan/prefill_decode/all_now",
            multistage(StageList::prefill_decode(), &decode, false),
            true,
        ),
        planned(
            "plan/prefill_decode_deferred",
            multistage(
                StageList::prefill_decode_deferred(Duration::from_secs(5)),
                &decode,
                false,
            ),
            false,
        ),
        planned(
            "plan/decode_first",
            multistage(StageList::decode_first(), &decode, false),
            false,
        ),
        planned(
            "plan/conditional/remote",
            multistage(conditional_list.clone(), &decode, true),
            false,
        ),
        planned(
            "plan/conditional/bypass",
            multistage(conditional_list, &cached_decode, true),
            false,
        ),
    ]
}

fn bench_routes(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let workers = workers_per_set();
    let mut group = c.benchmark_group("router_plan/route_then_release");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    for mut scenario in scenarios(&runtime, workers) {
        let mut sequence = 0u64;
        group.bench_function(BenchmarkId::new(&scenario.name, workers), |b| {
            b.iter(|| {
                sequence += 1;
                (scenario.run)(&runtime, sequence);
            });
        });
    }
    group.finish();
}

fn percentiles(samples: usize) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let workers = workers_per_set();
    let warmup = samples / 10;
    println!("scenario\tworkers_per_set\tsamples\tp50_us\tp95_us\tp99_us\tmean_us\tmax_us");
    for mut scenario in scenarios(&runtime, workers) {
        let mut durations = Vec::with_capacity(samples);
        for sequence in 0..(warmup + samples) as u64 {
            let started = Instant::now();
            (scenario.run)(&runtime, sequence);
            if sequence >= warmup as u64 {
                durations.push(started.elapsed());
            }
        }
        durations.sort_unstable();
        let at =
            |q: f64| durations[((durations.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
        let mean =
            durations.iter().map(Duration::as_secs_f64).sum::<f64>() / durations.len() as f64 * 1e6;
        println!(
            "{}\t{workers}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}",
            scenario.name,
            durations.len(),
            at(0.50),
            at(0.95),
            at(0.99),
            mean,
            durations.last().expect("samples").as_secs_f64() * 1e6
        );
    }
}

fn main() {
    if let Some(samples) = std::env::var("ROUTER_BENCH_PERCENTILES")
        .ok()
        .and_then(|value| value.parse().ok())
    {
        percentiles(samples);
        return;
    }
    let mut criterion = Criterion::default().configure_from_args();
    bench_routes(&mut criterion);
    criterion.final_summary();
}
