"""Phase 2 CLI: train 3 models (with hyperparameter tuning) -> select best on val
-> predict -> evaluate on test.

Usage (from project root):
    python -m phase2.src.pipeline_run --run-id v1
    python -m phase2.src.pipeline_run --run-id v1 --no-tune
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2.src.data import load_and_split
from phase2.src.evaluate import evaluate_all
from phase2.src.models import BEST_VERSION, MODEL_VERSIONS
from phase2.src.predict import write_all_predictions
from phase2.src.train import train_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _print_comparison(comparison: dict, eval_summary: dict, best_source: str) -> None:
    tuned = comparison.get("tuned", False)
    print(f"\n=== Validation comparison (selection){' — tuned' if tuned else ''} ===")
    print(f"{'model':<22} {'val_acc':>10} {'val_f1_macro':>14} {'val_f1_wt':>12}")
    for name, m in comparison["models"].items():
        marker = " <-- BEST" if name == best_source else ""
        print(
            f"{name:<22} {m['val_accuracy']:10.4f} {m['val_f1_macro']:14.4f} "
            f"{m['val_f1_weighted']:12.4f}{marker}"
        )
        if tuned and m.get("best_params"):
            print(f"  best_params: {m['best_params']}")
            if "cv_f1_macro" in m:
                print(f"  cv_f1_macro: {m['cv_f1_macro']:.4f}")

    print("\n=== Test metrics ===")
    print(f"{'model':<22} {'test_acc':>10} {'test_f1_macro':>14} {'is_best':>8}")
    for name, block in eval_summary["models"].items():
        m = block["metrics"]
        print(
            f"{name:<22} {m['accuracy']:10.4f} {m['f1_macro']:14.4f} "
            f"{int(m['is_best']):8d}"
        )
    print(f"\nWinner (validation f1_macro): {best_source}  alias={BEST_VERSION}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 ML pipeline (3 models)")
    parser.add_argument("--run-id", default="v1", help="Run / artifact suffix (default: v1)")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Split random seed"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip GridSearchCV and train with default hyperparameters",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Stratified CV folds for hyperparameter tuning (default: 3)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV (default: -1 = all cores)",
    )
    args = parser.parse_args(argv)

    tune = not args.no_tune
    logger.info(
        "Phase 2 pipeline starting (run_id=%s, tune=%s)", args.run_id, tune
    )
    split = load_and_split(random_state=args.random_state, run_id=args.run_id)

    train_result = train_all(
        split,
        run_id=args.run_id,
        tune=tune,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )
    pipelines = train_result["pipelines"]
    best_source = train_result["best_model_version"]
    comparison = train_result["comparison"]

    versions = list(MODEL_VERSIONS) + [BEST_VERSION]
    counts = write_all_predictions(pipelines, split.posts, versions)
    for v, n in counts.items():
        logger.info("Predictions stored: %s -> %s rows", v, n)

    eval_summary = evaluate_all(
        pipelines,
        split,
        best_source_version=best_source,
        val_metrics_by_model=comparison["models"],
        run_id=args.run_id,
    )

    _print_comparison(comparison, eval_summary, best_source)
    logger.info("Phase 2 pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
