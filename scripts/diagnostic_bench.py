import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "torch is required to run this diagnostic bench. "
        "Install deps from requirements.txt into your venv. "
        f"Import error: {e}"
    )

# Allow running as `python scripts/diagnostic_bench.py` from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, _REPO_ROOT)

from src.ml.maml_scheduler import SchedulingMAML  # noqa: E402
from src.nlp.deliverable_mapper import DeliverableMapper  # noqa: E402


@dataclass(frozen=True)
class BenchConfig:
    users: int
    support_events: int
    predict_iters: int
    warmup_iters: int
    seed: int
    preferred_hour: Optional[int]


def _utc_iso_at_hour(day_offset: int, hour: int) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    dt = base + timedelta(days=day_offset)
    dt = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    # Model code handles 'Z' specially; produce that for compatibility.
    return dt.isoformat().replace("+00:00", "Z")


def _percentiles_ms(samples_ms: List[float], ps: List[int]) -> Dict[str, float]:
    if not samples_ms:
        return {f"p{p}": math.nan for p in ps}
    arr = np.array(samples_ms, dtype=np.float64)
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def _prob_at_hour(recommendations: List[Dict[str, Any]], hour: int) -> float:
    for r in recommendations:
        if int(r.get("hour", -1)) == hour:
            return float(r.get("probability", 0.0))
    return 0.0


async def _embed_texts_offline(
    mapper: DeliverableMapper, texts: List[str]
) -> torch.Tensor:
    """
    Force the hashing fallback to avoid any model download attempts.
    This keeps the bench runnable in offline/sandboxed environments.
    """
    mapper.use_fallback = True
    mapper.model = None
    results = await mapper.batch_map_deliverables(texts, contexts=[{} for _ in texts])
    return torch.tensor([r["embedding"] for r in results], dtype=torch.float32)


def _make_support_events(
    user_id: str, preferred_hour: int, support_events: int, deliverable_texts: List[str]
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for i in range(support_events):
        events.append(
            {
                "deliverable": {
                    "title": deliverable_texts[i],
                    "priority": "medium",
                    "urgency": "medium",
                },
                "context": {
                    # Put non-time context here; the model injects timestamp from start.dateTime
                    "source": "diagnostic_bench",
                },
                "start": {
                    "dateTime": _utc_iso_at_hour(
                        day_offset=(hash(user_id) + i) % 28, hour=preferred_hour
                    )
                },
            }
        )
    return events


async def run_bench(cfg: BenchConfig) -> Dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    mapper = DeliverableMapper()
    model = SchedulingMAML()

    users = []
    for u in range(cfg.users):
        user_id = f"user_{u:04d}"
        preferred_hour = (
            cfg.preferred_hour if cfg.preferred_hour is not None else (u % 24)
        )
        users.append((user_id, preferred_hour))

    # --- Warmup (helps stabilize measurements) ---
    warmup_lat_ms: List[float] = []
    for i in range(cfg.warmup_iters):
        user_id, hour = users[i % len(users)]
        query_ctx = {"timestamp": _utc_iso_at_hour(day_offset=999, hour=hour)}
        deliverable = {"title": f"warmup {i}", "embedding": [0.0] * 384}
        t0 = time.perf_counter_ns()
        await model.predict_optimal_slots(
            user_id=user_id, deliverable=deliverable, context=query_ctx
        )
        t1 = time.perf_counter_ns()
        warmup_lat_ms.append((t1 - t0) / 1e6)

    # --- Per-user adaptation + sanity metrics (before/after) ---
    pre_top1_hits = 0
    post_top1_hits = 0
    pre_prob_target: List[float] = []
    post_prob_target: List[float] = []
    adapt_ms: List[float] = []

    for user_id, preferred_hour in users:
        # Create support set
        support_texts = [
            f"{user_id} task {i}: prepare report and follow up"
            for i in range(cfg.support_events)
        ]
        support_events = _make_support_events(
            user_id=user_id,
            preferred_hour=preferred_hour,
            support_events=cfg.support_events,
            deliverable_texts=support_texts,
        )

        support_embeddings = await _embed_texts_offline(mapper, support_texts)

        # Query deliverable embedding (offline)
        query_text = f"{user_id} query: implement feature and write notes"
        query_emb = (await _embed_texts_offline(mapper, [query_text]))[0].tolist()

        # IMPORTANT: this model uses time features derived from context['timestamp'].
        # To evaluate *learnability* from a user’s historical scheduled hour,
        # we set query timestamp to the preferred hour and check probability mass/top-1
        # before vs after user-specific adaptation.
        query_ctx = {
            "timestamp": _utc_iso_at_hour(day_offset=1234, hour=preferred_hour)
        }
        query_deliverable = {
            "title": query_text,
            "priority": "medium",
            "urgency": "medium",
            "embedding": query_emb,
        }

        pre_recs = await model.predict_optimal_slots(
            user_id=user_id, deliverable=query_deliverable, context=query_ctx
        )
        if pre_recs:
            pre_top1_hits += int(int(pre_recs[0]["hour"]) == preferred_hour)
            pre_prob_target.append(_prob_at_hour(pre_recs, preferred_hour))

        t0 = time.perf_counter_ns()
        await model.adapt_to_user(
            user_id=user_id,
            support_events=support_events,
            deliverable_embeddings=support_embeddings,
        )
        t1 = time.perf_counter_ns()
        adapt_ms.append((t1 - t0) / 1e6)

        post_recs = await model.predict_optimal_slots(
            user_id=user_id, deliverable=query_deliverable, context=query_ctx
        )
        if post_recs:
            post_top1_hits += int(int(post_recs[0]["hour"]) == preferred_hour)
            post_prob_target.append(_prob_at_hour(post_recs, preferred_hour))

    # --- Prediction latency sampling (post-adaptation; mixed users) ---
    pred_lat_ms: List[float] = []
    for i in range(cfg.predict_iters):
        user_id, preferred_hour = users[i % len(users)]
        query_ctx = {"timestamp": _utc_iso_at_hour(day_offset=555, hour=preferred_hour)}
        deliverable = {"title": f"bench {i}", "embedding": [0.0] * 384}
        t0 = time.perf_counter_ns()
        await model.predict_optimal_slots(
            user_id=user_id, deliverable=deliverable, context=query_ctx
        )
        t1 = time.perf_counter_ns()
        pred_lat_ms.append((t1 - t0) / 1e6)

    summary: Dict[str, Any] = {
        "config": {
            "users": cfg.users,
            "support_events": cfg.support_events,
            "predict_iters": cfg.predict_iters,
            "warmup_iters": cfg.warmup_iters,
            "seed": cfg.seed,
            "preferred_hour": cfg.preferred_hour,
            "device": str(getattr(model, "device", "unknown")),
        },
        "sanity": {
            "top1_pre": pre_top1_hits / cfg.users if cfg.users else math.nan,
            "top1_post": post_top1_hits / cfg.users if cfg.users else math.nan,
            "prob_pre_mean": (
                float(statistics.fmean(pre_prob_target))
                if pre_prob_target
                else math.nan
            ),
            "prob_post_mean": (
                float(statistics.fmean(post_prob_target))
                if post_prob_target
                else math.nan
            ),
        },
        "timing_ms": {
            "adapt": {
                "mean": float(statistics.fmean(adapt_ms)) if adapt_ms else math.nan,
                **_percentiles_ms(adapt_ms, [50, 95, 99]),
            },
            "predict": {
                "mean": (
                    float(statistics.fmean(pred_lat_ms)) if pred_lat_ms else math.nan
                ),
                **_percentiles_ms(pred_lat_ms, [50, 95, 99]),
            },
            "warmup_predict": {
                "mean": (
                    float(statistics.fmean(warmup_lat_ms))
                    if warmup_lat_ms
                    else math.nan
                ),
                **_percentiles_ms(warmup_lat_ms, [50, 95, 99]),
            },
        },
    }

    # Human-readable output (then JSON blob)
    print("\n=== EnginEdge Scheduling Model Diagnostic Bench ===")
    print(
        f"users={cfg.users} support_events={cfg.support_events} predict_iters={cfg.predict_iters}"
    )
    print(f"device={summary['config']['device']}")
    print("\nSanity metrics (synthetic):")
    print(f"  top1_pre  = {summary['sanity']['top1_pre']:.3f}")
    print(f"  top1_post = {summary['sanity']['top1_post']:.3f}")
    print(f"  prob_pre_mean  = {summary['sanity']['prob_pre_mean']:.4f}")
    print(f"  prob_post_mean = {summary['sanity']['prob_post_mean']:.4f}")
    print("\nLatency (ms):")
    print(
        "  predict: "
        f"mean={summary['timing_ms']['predict']['mean']:.3f} "
        f"p50={summary['timing_ms']['predict']['p50']:.3f} "
        f"p95={summary['timing_ms']['predict']['p95']:.3f} "
        f"p99={summary['timing_ms']['predict']['p99']:.3f}"
    )
    print(
        "  adapt:   "
        f"mean={summary['timing_ms']['adapt']['mean']:.3f} "
        f"p50={summary['timing_ms']['adapt']['p50']:.3f} "
        f"p95={summary['timing_ms']['adapt']['p95']:.3f} "
        f"p99={summary['timing_ms']['adapt']['p99']:.3f}"
    )

    print("\nJSON:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return summary


def parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(
        description="Offline diagnostic benchmark for SchedulingMAML"
    )
    p.add_argument("--users", type=int, default=50)
    p.add_argument("--support-events", type=int, default=5)
    p.add_argument("--predict-iters", type=int, default=2000)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--preferred-hour",
        type=int,
        default=None,
        help=(
            "If set, all users share the same preferred hour (0-23). "
            "Otherwise user_i uses i % 24."
        ),
    )
    args = p.parse_args()
    if args.users <= 0:
        raise SystemExit("--users must be > 0")
    if args.support_events <= 0:
        raise SystemExit("--support-events must be > 0")
    if args.predict_iters <= 0:
        raise SystemExit("--predict-iters must be > 0")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be >= 0")
    if args.preferred_hour is not None and not (0 <= args.preferred_hour <= 23):
        raise SystemExit("--preferred-hour must be in [0, 23]")
    return BenchConfig(
        users=args.users,
        support_events=args.support_events,
        predict_iters=args.predict_iters,
        warmup_iters=args.warmup_iters,
        seed=args.seed,
        preferred_hour=args.preferred_hour,
    )


def main() -> None:
    cfg = parse_args()
    asyncio.run(run_bench(cfg))


if __name__ == "__main__":
    main()
