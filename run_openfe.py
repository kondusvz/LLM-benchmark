"""Run OpenFE and save one full-dataset feature matrix per run."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset_utils import (
    DatasetLoader,
    DatasetValidator,
    FeatureCSVWriter,
    build_feature_output_path,
    get_next_feature_run_index,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def import_openfe():
    """Import OpenFE and return the ``OpenFE`` class."""
    try:
        import lightgbm as lgb
        from openfe import OpenFE
    except ImportError as exc:
        raise ImportError(
            "Could not import openfe. Install it with:\n"
            "  pip install openfe lightgbm\n"
            "Do not use 'conda install openfe'; that installs an unrelated package."
        ) from exc

    if not getattr(lgb, "_openfe_quiet_patch_applied", False):
        class _FilteredLightGBMLogger:
            def info(self, msg: str) -> None:
                text = str(msg).strip()
                if "No further splits with positive gain, best gain: -inf" in text:
                    return
                logger.info("[LightGBM] %s", text)

            def warning(self, msg: str) -> None:
                text = str(msg).strip()
                if "No further splits with positive gain, best gain: -inf" in text:
                    return
                logger.warning("[LightGBM] %s", text)

        lgb.register_logger(_FilteredLightGBMLogger())
        lgb._openfe_quiet_patch_applied = True
        logger.info("Applied LightGBM log filter for OpenFE.")

    logger.info("OpenFE imported successfully.")
    return OpenFE


def load_and_prepare(
    loader: DatasetLoader,
    dataset_name: str,
    datasets_dir: str,
) -> tuple[pd.DataFrame, pd.Series, str, list[str]]:
    """
    Load one dataset using a cleaned raw DataFrame so OpenFE can retain its
    own categorical handling. A numeric-only export is created later.
    """
    df, target_col, task_type = loader.read_and_clean(dataset_name, datasets_dir)
    cat_map = loader.detect_categorical(df, target_col, dataset_name=dataset_name)
    categorical_features = [col for col, is_cat in cat_map.items() if is_cat]
    X_df = df.drop(columns=[target_col]).copy()
    y_series = df[target_col].copy()
    logger.info(
        "Loaded '%s': %d rows x %d cleaned raw features (%d categorical), task=%s",
        dataset_name,
        X_df.shape[0],
        X_df.shape[1],
        len(categorical_features),
        task_type,
    )
    return X_df, y_series, task_type, categorical_features


def _encode_features_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the final OpenFE matrix to a numeric-only frame for the benchmark
    evaluator while preserving raw categorical handling during feature search.
    """
    encoded = df.copy()

    for col in encoded.columns:
        series = encoded[col]
        if pd.api.types.is_bool_dtype(series):
            encoded[col] = series.astype(int)
        elif pd.api.types.is_numeric_dtype(series):
            encoded[col] = pd.to_numeric(series, errors="coerce")
        else:
            label_encoder = LabelEncoder()
            values = series.astype("string").fillna("__missing__")
            encoded[col] = label_encoder.fit_transform(values)

    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0)
    return encoded


def _make_safe_feature_names(columns: list[str]) -> dict[str, str]:
    """
    OpenFE reparses feature formulas, so raw names containing operators like
    '-' can break during stage2/transform. Map base columns to safe identifiers
    for internal OpenFE use.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for idx, col in enumerate(columns):
        safe = re.sub(r"[^0-9A-Za-z_]", "_", str(col))
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = f"feature_{idx}"
        if safe[0].isdigit():
            safe = f"f_{safe}"

        candidate = safe
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{safe}_{suffix}"

        mapping[col] = candidate
        used.add(candidate)

    return mapping


def run_openfe(
    OpenFE,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    categorical_features: list[str],
    *,
    n_jobs: int = 1,
    top_k: int = 10,
    n_data_blocks: int = 8,
    feature_boosting: bool = True,
    seed: int = 0,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Fit OpenFE on a train split and return a full-dataset feature matrix."""
    task_str = "regression" if task_type == "regression" else "classification"
    n_classes = int(y.nunique()) if task_type == "classification" else None
    effective_n_data_blocks = n_data_blocks
    if task_type == "classification" and n_classes and n_classes > 2 and n_data_blocks != 1:
        effective_n_data_blocks = 1
        logger.info(
            "Using n_data_blocks=1 instead of %d for multi-class classification "
            "(%d classes) to avoid OpenFE stage1 init-score class mismatches.",
            n_data_blocks,
            n_classes,
        )

    logger.info(
        "Running OpenFE: top_k=%d, n_jobs=%d, n_data_blocks=%d, "
        "feature_boosting=%s, task=%s, test_size=%.2f, seed=%d",
        top_k,
        n_jobs,
        effective_n_data_blocks,
        feature_boosting,
        task_str,
        test_size,
        seed,
    )

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    logger.info(
        "Train/test split: %d train rows, %d test rows.",
        len(X_train),
        len(X_test),
    )

    rename_map = _make_safe_feature_names(list(X.columns))
    inverse_rename_map = {safe: original for original, safe in rename_map.items()}
    categorical_features_safe = [rename_map[col] for col in categorical_features]

    X_train_safe = X_train.rename(columns=rename_map)
    X_test_safe = X_test.rename(columns=rename_map)

    ofe = OpenFE()
    features = ofe.fit(
        data=X_train_safe,
        label=y_train.to_frame(name=y.name or "target"),
        n_jobs=n_jobs,
        n_data_blocks=effective_n_data_blocks,
        feature_boosting=feature_boosting,
        task=task_str,
        categorical_features=categorical_features_safe,
        seed=seed,
    )
    logger.info("OpenFE returned %d ranked feature(s).", len(features))

    selected = features[:top_k]
    logger.info("Selecting top %d features.", len(selected))

    X_train_aug, X_test_aug = ofe.transform(
        X_train_safe.copy(),
        X_test_safe.copy(),
        selected,
        n_jobs=n_jobs,
    )

    X_train_aug = X_train_aug.rename(columns=inverse_rename_map)
    X_test_aug = X_test_aug.rename(columns=inverse_rename_map)

    new_cols = [col for col in X_train_aug.columns if col not in X.columns]
    logger.info("New columns added: %d", len(new_cols))

    X_full_aug = pd.concat([X_train_aug, X_test_aug]).reindex(X.index)
    X_full_aug = X_full_aug.replace([np.inf, -np.inf], np.nan).fillna(0)

    const_mask = X_full_aug.nunique() <= 1
    if const_mask.any():
        dropped = list(X_full_aug.columns[const_mask])
        logger.warning("Dropping %d constant column(s): %s", len(dropped), dropped)
        X_full_aug = X_full_aug.loc[:, ~const_mask]

    X_full_aug = X_full_aug.T.drop_duplicates().T

    X_full_aug = _encode_features_for_storage(X_full_aug)

    logger.info(
        "Augmented matrix shape: %d rows x %d columns (%d original + %d new)",
        X_full_aug.shape[0],
        X_full_aug.shape[1],
        X.shape[1],
        X_full_aug.shape[1] - X.shape[1],
    )
    return X_full_aug


def _write_meta(
    output_path: str,
    dataset: str,
    runtime_seconds: float,
    n_original: int,
    n_final: int,
    run_index: int,
    seed: int,
    top_k: int,
    n_data_blocks: int,
    feature_boosting: bool,
    n_jobs: int,
    test_size: float,
) -> None:
    """Write one metadata sidecar next to the feature CSV."""
    meta = {
        "dataset": dataset,
        "method": "openfe",
        "run_index": run_index,
        "seed": seed,
        "runtime_seconds": round(runtime_seconds, 2),
        "n_original_features": n_original,
        "n_final_features": n_final,
        "n_added_features": n_final - n_original,
        "top_k": top_k,
        "n_data_blocks": n_data_blocks,
        "feature_boosting": feature_boosting,
        "n_jobs": n_jobs,
        "test_size": test_size,
    }
    meta_path = output_path.replace("_features.csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Metadata written: %s", meta_path)


def _parse_dataset_names(values) -> list[str]:
    """Normalize dataset lists from CLI input."""
    if not values:
        return []

    names: list[str] = []
    for value in values:
        for part in str(value).split(","):
            name = part.strip()
            if name:
                names.append(name)
    return names


def _resolve_datasets(args, loader: DatasetLoader) -> list[str]:
    """Resolve the dataset list after applying optional skips."""
    skip_datasets = set(_parse_dataset_names(args.skip_datasets))

    if args.all_datasets:
        datasets = loader.list_datasets()
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise ValueError("Provide --dataset or use --all-datasets.")

    if skip_datasets:
        datasets = [dataset for dataset in datasets if dataset not in skip_datasets]

    if not datasets:
        raise ValueError("No datasets selected after applying --skip-datasets.")

    unknown_skips = sorted(skip_datasets.difference(loader.list_datasets()))
    if unknown_skips:
        logger.warning("Ignoring unknown dataset name(s) in --skip-datasets: %s", unknown_skips)

    return datasets


def _run_single_dataset(
    args,
    loader: DatasetLoader,
    validator: DatasetValidator,
    OpenFE,
    dataset_name: str,
    method: str,
    run_index: int,
    run_seed: int,
) -> str:
    """Run OpenFE once for one dataset and save the result."""
    feature_boosting = not args.no_feature_boosting
    logger.info(
        "=== OpenFE | dataset=%s | run=%s-%d | top_k=%d | n_jobs=%d | "
        "n_data_blocks=%d | feature_boosting=%s | test_size=%.2f | seed=%d ===",
        dataset_name,
        dataset_name,
        run_index,
        args.top_k,
        args.n_jobs,
        args.n_data_blocks,
        feature_boosting,
        args.test_size,
        run_seed,
    )

    X, y, task_type, categorical_features = load_and_prepare(loader, dataset_name, args.datasets_dir)
    report = validator.validate(X, y, task_type, dataset_name)
    if not report["is_valid"]:
        raise ValueError(f"Dataset validation failed: {report['warnings']}")

    n_original = X.shape[1]
    started = time.time()
    X_augmented = run_openfe(
        OpenFE,
        X,
        y,
        task_type,
        categorical_features,
        n_jobs=args.n_jobs,
        top_k=args.top_k,
        n_data_blocks=args.n_data_blocks,
        feature_boosting=feature_boosting,
        seed=run_seed,
        test_size=args.test_size,
    )
    runtime = time.time() - started

    out_path = str(
        build_feature_output_path(
            args.output_dir,
            method,
            dataset_name,
            run_index,
        )
    )
    FeatureCSVWriter().write(X_augmented, out_path)
    _write_meta(
        out_path,
        dataset=dataset_name,
        runtime_seconds=runtime,
        n_original=n_original,
        n_final=X_augmented.shape[1],
        run_index=run_index,
        seed=run_seed,
        top_k=args.top_k,
        n_data_blocks=args.n_data_blocks,
        feature_boosting=feature_boosting,
        n_jobs=args.n_jobs,
        test_size=args.test_size,
    )

    logger.info(
        "=== Done | dataset=%s | run=%s-%d | features=%d->%d | runtime=%.1fs | saved to %s ===",
        dataset_name,
        dataset_name,
        run_index,
        n_original,
        X_augmented.shape[1],
        runtime,
        out_path,
    )
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OpenFE and save the resulting feature matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--loop-iterations", type=int, default=1)
    parser.add_argument(
        "--skip-datasets",
        nargs="*",
        default=[],
        help="Dataset name(s) to skip. Accepts space-separated values or comma-separated lists.",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_data_blocks", type=int, default=8)
    parser.add_argument("--no_feature_boosting", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("--output_dir", default="./features")
    parser.add_argument("--method", default="openfe")
    args = parser.parse_args()

    if args.loop_iterations < 1:
        parser.error("--loop-iterations must be >= 1")

    OpenFE = import_openfe()
    loader = DatasetLoader()
    validator = DatasetValidator()
    datasets = _resolve_datasets(args, loader)

    logger.info(
        "Output method directory: %s | run plan: %d dataset(s) x %d loop iteration(s)",
        args.method,
        len(datasets),
        args.loop_iterations,
    )

    failures: list[str] = []
    for dataset_name in datasets:
        next_run_index = get_next_feature_run_index(args.output_dir, args.method, dataset_name)
        for offset in range(args.loop_iterations):
            run_index = next_run_index + offset
            run_seed = args.seed + offset
            try:
                _run_single_dataset(
                    args,
                    loader,
                    validator,
                    OpenFE,
                    dataset_name,
                    args.method,
                    run_index,
                    run_seed,
                )
            except Exception as exc:
                logger.exception(
                    "Run failed for dataset='%s' run='%s-%d': %s",
                    dataset_name,
                    dataset_name,
                    run_index,
                    exc,
                )
                failures.append(f"{dataset_name}-{run_index}: {exc}")
                if not args.all_datasets:
                    raise

    if failures:
        logger.error("Completed with %d failed run(s).", len(failures))
        for failure in failures:
            logger.error("  %s", failure)
        sys.exit(1)


if __name__ == "__main__":
    main()
