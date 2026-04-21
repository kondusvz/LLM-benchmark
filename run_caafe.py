"""Run the benchmark-integrated CAAFE wrapper and save one feature matrix per run."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_utils import (
    DatasetLoader,
    FeatureCSVWriter,
    build_feature_output_path,
    get_next_feature_run_index,
)


def _model_slug(model: str) -> str:
    """Convert a model identifier into a filesystem-safe slug, stripping provider prefix."""
    slug = model.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", slug)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
DEFAULT_API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _usage_attr(obj, attr, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _response_attr(obj, attr, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _extract_request_id(response):
    request_id = _response_attr(response, "id")
    x_groq = _response_attr(response, "x_groq")
    if x_groq is not None:
        request_id = _response_attr(x_groq, "id", request_id)
    return request_id


def _extract_token_usage(response, call_index: int, elapsed_seconds: float) -> dict:
    usage = _response_attr(response, "usage")
    input_tokens = _usage_attr(usage, "prompt_tokens", None)
    output_tokens = _usage_attr(usage, "completion_tokens", None)
    total_tokens = _usage_attr(usage, "total_tokens", None)

    if input_tokens is None:
        input_tokens = _usage_attr(usage, "input_tokens", 0)
    if output_tokens is None:
        output_tokens = _usage_attr(usage, "output_tokens", 0)
    if total_tokens is None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return {
        "call_index": call_index,
        "request_id": _extract_request_id(response),
        "response_created": _response_attr(response, "created"),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def _summarise_token_usage(llm_calls: list[dict]) -> dict:
    return {
        "llm_call_count": len(llm_calls),
        "input_tokens": sum(call.get("input_tokens", 0) for call in llm_calls),
        "output_tokens": sum(call.get("output_tokens", 0) for call in llm_calls),
        "total_tokens": sum(call.get("total_tokens", 0) for call in llm_calls),
        "calls": llm_calls,
    }


try:
    import tabpfn.scripts as _tabpfn_scripts
    if not hasattr(_tabpfn_scripts, "tabular_metrics"):
        from tabpfn.scripts import tabular_metrics as _tabular_metrics
        _tabpfn_scripts.tabular_metrics = _tabular_metrics
        logger.info("Injected tabpfn.scripts.tabular_metrics (TabPFN compatibility patch).")
except Exception as _e:
    logger.warning("Could not apply TabPFN compatibility patch: %s", _e)


def build_base_classifier():
    """
    Attempt to import and return a TabPFN v1 classifier.

    TabPFN v1 is the base classifier used in the CAAFE paper.  If it is not
    installed or fails to import, fall back to RandomForestClassifier and log
    a warning so the deviation from the paper setup is visible in logs.

    Returns
    -------
    clf : sklearn-compatible classifier
    clf_name : str
        Human-readable name for logging.
    """
    try:
        from functools import partial
        import torch
        from tabpfn import TabPFNClassifier  # tabpfn<2 expected

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf = TabPFNClassifier(device=device, N_ensemble_configurations=16)
        # Suppress the "overwrite_warning" prompt that TabPFN v1 raises when
        # fit() is called more than once on the same object (happens inside
        # CAAFE's CV loop).
        clf.fit = partial(clf.fit, overwrite_warning=True)
        logger.info(
            "Base classifier: TabPFNClassifier (device=%s, ensembles=%d)",
            device,
            16,
        )
        return clf, "TabPFNClassifier"
    except Exception as exc:
        logger.warning(
            "TabPFN could not be imported (%s). "
            "Falling back to RandomForestClassifier.  "
            "Results will deviate from the paper's setup.",
            exc,
        )
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
        return clf, "RandomForestClassifier (fallback)"


def setup_environment() -> tuple[str, str]:
    """
    Validate the OpenAI API key and resolve the model name.

    CAAFE reads openai.api_key and openai.api_base from the module-level
    globals (pre-1.0 SDK style).  These are set here so that CAAFE's
    internal LLM calls pick them up automatically.

    Returns
    -------
    api_key : str
    model : str
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY=your_key_here"
        )

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    # Set module-level globals on the (old) openai package that caafe imports.
    try:
        import openai as _openai
        _openai.api_key = api_key
        _openai.api_base = base_url
    except Exception as exc:
        logger.warning("Could not configure openai globals: %s", exc)

    logger.info("API key : %s...%s", api_key[:8], api_key[-4:])
    logger.info("API base: %s", base_url)
    logger.info("Model   : %s", model)
    return api_key, model


def prepare_dataframes(
    loader: DatasetLoader,
    dataset_name: str,
    datasets_dir: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load the dataset and produce train/test DataFrames suitable for CAAFE.

    CAAFE's internal ``get_X_y`` function calls ``target.astype(int)``, so the
    target column must be integer-encoded before being passed to ``fit_pandas``.
    Feature columns are left as-is (strings preserved) so the LLM prompt
    receives readable column values for domain-aware feature generation.

    The same LabelEncoder fitted on the full dataset is applied to both splits
    so that class indices are consistent across train and test.

    Parameters
    ----------
    loader : DatasetLoader
    dataset_name : str
    datasets_dir : str
    test_size : float
    seed : int

    Returns
    -------
    df_full : pd.DataFrame
        Full cleaned dataset with integer-encoded target as last column.
    df_train : pd.DataFrame
        Training split with integer-encoded target as last column.
    df_test : pd.DataFrame
        Test split with integer-encoded target as last column.
    target_col : str
    task_type : str
    """
    from sklearn.preprocessing import LabelEncoder

    df, target_col, task_type = loader.read_and_clean(dataset_name, datasets_dir)

    # Encode target to integers - required by CAAFE's internal get_X_y().
    # Features are intentionally left un-encoded so the LLM prompt sees
    # human-readable values (e.g. 'Iris-setosa' rather than 0).
    if task_type == "classification":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        logger.info(
            "Target '%s' label-encoded: %s",
            target_col, list(enumerate(le.classes_)),
        )

    # Stratified split - target is the last column after read_and_clean.
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    logger.info(
        "Split '%s': %d train / %d test rows, target='%s'",
        dataset_name, len(df_train), len(df_test), target_col,
    )
    return df, df_train, df_test, target_col, task_type


def run_caafe(
    df_fit: pd.DataFrame,
    df_full: pd.DataFrame,
    target_col: str,
    dataset_description: str,
    base_clf,
    model: str,
    iterations: int,
    n_splits: int = 10,
    n_repeats: int = 2,
    llm_calls=None,
) -> tuple[pd.DataFrame, str]:
    """
    Fit a CAAFEClassifier and apply the learned feature transformations
    to the full dataset.

    CAAFE iteratively prompts the LLM to generate Python feature-engineering
    code, executes it, validates each new feature via cross-validation on the
    fit data, and retains only features that improve performance. After
    fitting, ``caafe_clf.code`` contains the accepted transformation code.

    Parameters
    ----------
    df_fit : pd.DataFrame
        Data used by CAAFE during feature generation, including target as
        the last column.
    df_full : pd.DataFrame
        Full cleaned dataset including target as last column.
    target_col : str
    dataset_description : str
        Natural-language description passed to the LLM.
    base_clf : sklearn-compatible classifier
    model : str
        OpenAI model name.
    iterations : int
        Number of CAAFE iterations (LLM calls + CV validations).
    n_splits : int
        Number of CV folds per iteration.
    n_repeats : int
        Number of CV repetitions per iteration.

    Returns
    -------
    df_full_augmented : pd.DataFrame
        Full dataset with original + CAAFE-generated features (target excluded).
    generated_code : str
        The accepted Python code string for transparency / logging.
    """
    from caafe import CAAFEClassifier

    # CAAFE's internal progress display uses IPython.display.display(), which
    # in a terminal context prints the repr of Markdown objects endlessly and
    # can cause an apparent infinite loop.  We replace the display function
    # with a no-op before fitting and restore it afterwards.
    try:
        import IPython.display as _ipython_display
        _original_display = _ipython_display.display
        _ipython_display.display = lambda *args, **kwargs: None
        _patched_display = True
    except ImportError:
        _patched_display = False

    try:
        try:
            import openai as _openai
            _original_chat_create = _openai.ChatCompletion.create

            def _logged_chat_create(*args, **kwargs):
                call_start = time.time()
                response = _original_chat_create(*args, **kwargs)
                if llm_calls is not None:
                    llm_calls.append(
                        _extract_token_usage(
                            response,
                            call_index=len(llm_calls) + 1,
                            elapsed_seconds=time.time() - call_start,
                        )
                    )
                return response

            _openai.ChatCompletion.create = _logged_chat_create
            _patched_openai = True
        except Exception as exc:
            logger.warning("Could not patch OpenAI token logging for CAAFE: %s", exc)
            _patched_openai = False
            _openai = None
            _original_chat_create = None

        caafe_clf = CAAFEClassifier(
            base_classifier=base_clf,
            llm_model=model,
            iterations=iterations,
            n_splits=n_splits,
            n_repeats=n_repeats,
        )

        logger.info(
            "Fitting CAAFE (model=%s, iterations=%d, base=%s, n_splits=%d, n_repeats=%d)...",
            model, iterations, type(base_clf).__name__, n_splits, n_repeats,
        )

        caafe_clf.fit_pandas(
            df_fit,
            target_column_name=target_col,
            dataset_description=dataset_description,
        )

    finally:
        if _patched_openai and _openai is not None:
            _openai.ChatCompletion.create = _original_chat_create
        # Restore IPython display unconditionally so the process is never left
        # in a broken state regardless of whether fit_pandas succeeded.
        if _patched_display:
            _ipython_display.display = _original_display

    generated_code = caafe_clf.code or ""
    n_new = generated_code.count("def ") if generated_code else 0
    logger.info(
        "CAAFE fitting complete. Accepted transformation blocks: ~%d", n_new
    )
    if generated_code:
        logger.info("Generated code:\n%s", generated_code)
    else:
        logger.warning(
            "CAAFE produced no accepted features for this dataset.  "
            "The output will contain only the original features."
        )

    # Apply the learned transformations to the full dataset so the saved
    # feature matrix stays aligned with evaluator labels row-by-row.
    # CAAFEClassifier stores the code as a string; we execute it against
    # a copy of df_full (minus the target) to obtain the augmented features.
    X_full = df_full.drop(columns=[target_col]).copy()

    if generated_code:
        try:
            # CAAFE's internal apply helper executes the stored code safely.
            # We replicate its approach: exec the code in a namespace that
            # has pandas and numpy available, then call the transform.
            # CAAFE generates code that mutates a DataFrame named `df` inline,
            # e.g. `df['new_col'] = df['a'] / df['b']`.  We pass X_full into
            # the exec namespace as `df` and read it back after execution.
            X_full_copy = X_full.copy()
            ns: dict = {"pd": pd, "np": np, "df": X_full_copy}
            exec(generated_code, ns)  # noqa: S102
            X_full_aug = ns["df"]

            # Ensure all columns are numeric - CAAFE may produce object cols
            # if the LLM generates string-valued or boolean features.
            for col in X_full_aug.columns:
                if not pd.api.types.is_numeric_dtype(X_full_aug[col]):
                    try:
                        X_full_aug[col] = pd.to_numeric(
                            X_full_aug[col], errors="coerce"
                        ).fillna(0)
                    except Exception:
                        X_full_aug = X_full_aug.drop(columns=[col])

            logger.info(
                "Full-dataset features after augmentation: %d columns "
                "(was %d before CAAFE)",
                X_full_aug.shape[1], X_full.shape[1],
            )
            return X_full_aug, generated_code
        except Exception as exc:
            logger.warning(
                "Failed to apply generated code to full dataset (%s: %s). "
                "Returning original features only.",
                type(exc).__name__, exc,
            )
            logger.debug(traceback.format_exc())

    # Fallback: return original full-dataset features only (label-encode for evaluator
    # compatibility, since read_and_clean preserves strings).
    from sklearn.preprocessing import LabelEncoder
    for col in X_full.columns:
        if not pd.api.types.is_numeric_dtype(X_full[col]):
            X_full[col] = LabelEncoder().fit_transform(X_full[col].astype(str))
    return X_full, generated_code


def _write_meta(
    output_path: str,
    dataset: str,
    model: str,
    runtime_seconds: float,
    n_original: int,
    n_final: int,
    run_index: int,
    seed: int,
    token_usage=None,
) -> None:
    meta = {
        "dataset": dataset,
        "model": model,
        "run_index": run_index,
        "seed": seed,
        "runtime_seconds": round(runtime_seconds, 2),
        "token_usage": token_usage or _summarise_token_usage([]),
        "n_original_features": n_original,
        "n_final_features": n_final,
        "n_added_features": n_final - n_original,
    }
    meta_path = output_path.replace("_features.csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Metadata written: %s", meta_path)


def _resolve_datasets(args, loader: DatasetLoader) -> list[str]:
    if args.all_datasets:
        return loader.list_datasets(task_type="classification")
    if args.dataset:
        return [args.dataset]
    raise ValueError("Provide --dataset or use --all-datasets.")


def _resolve_dataset_description(loader: DatasetLoader, dataset_name: str) -> str:
    dataset_description = loader.get_description(dataset_name)
    if dataset_description is None:
        logger.warning(
            "No description found for '%s' in DatasetLoader.DATASET_DESCRIPTIONS. "
            "CAAFE will have limited domain context.  Consider adding a description.",
            dataset_name,
        )
        return (
            f"Tabular classification dataset named '{dataset_name}'. "
            f"Predict the target column from the available features."
        )

    logger.info("Dataset description loaded (%d chars).", len(dataset_description))
    return dataset_description


def _run_single_dataset(
    args,
    loader: DatasetLoader,
    dataset_name: str,
    method: str,
    model: str,
    run_index: int,
    run_seed: int,
) -> str:
    task_type = loader.get_task_type(dataset_name)
    if task_type == "regression":
        raise ValueError(
            f"Dataset '{dataset_name}' is a regression task. CAAFE only supports classification."
        )

    logger.info(
        "=== CAAFE | dataset=%s | run=%s-%d | iterations=%d | n_splits=%d | n_repeats=%d | seed=%d ===",
        dataset_name,
        dataset_name,
        run_index,
        args.iterations,
        args.n_splits,
        args.n_repeats,
        run_seed,
    )
    base_clf, clf_name = build_base_classifier()
    logger.info("Using base classifier: %s", clf_name)

    dataset_description = _resolve_dataset_description(loader, dataset_name)
    df_full, df_train, _df_test, target_col, _task_type = prepare_dataframes(
        loader,
        dataset_name,
        args.datasets_dir,
        args.test_size,
        run_seed,
    )
    df_fit = df_full if args.fit_mode == "full" else df_train
    logger.info(
        "CAAFE fit mode: %s (%d rows used for feature generation)",
        args.fit_mode,
        len(df_fit),
    )

    _t0 = time.time()
    llm_calls: list[dict] = []
    X_features, _generated_code = run_caafe(
        df_fit=df_fit,
        df_full=df_full,
        target_col=target_col,
        dataset_description=dataset_description,
        base_clf=base_clf,
        model=model,
        iterations=args.iterations,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        llm_calls=llm_calls,
    )

    out_path = str(
        build_feature_output_path(
            args.output_dir,
            method,
            dataset_name,
            run_index,
        )
    )
    FeatureCSVWriter().write(X_features, out_path)
    _write_meta(
        out_path,
        dataset_name,
        model,
        runtime_seconds=time.time() - _t0,
        n_original=len(df_full.columns) - 1,
        n_final=X_features.shape[1],
        run_index=run_index,
        seed=run_seed,
        token_usage=_summarise_token_usage(llm_calls),
    )
    logger.info(
        "=== Done | dataset=%s | run=%s-%d | features=%d | saved to %s ===",
        dataset_name,
        dataset_name,
        run_index,
        X_features.shape[1],
        out_path,
    )
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run CAAFE (Context-Aware Automated Feature Engineering) and save "
            "the augmented feature matrix for downstream evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default=None,
        help=(
            "Dataset name matching DatasetLoader.DATASET_METADATA "
            "(e.g. diabetes, breast-w, Titanic-Dataset)."
        ),
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Run sequentially over every classification dataset in DatasetLoader.DATASET_METADATA.",
    )
    parser.add_argument(
        "--loop-iterations", type=int, default=1,
        help="Number of full CAAFE runs to execute per dataset, saving each run separately.",
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of CAAFE iterations (LLM calls + CV validations).",
    )
    parser.add_argument(
        "--n_splits", type=int, default=10,
        help=(
            "Number of CV folds used internally by CAAFE to validate each "
            "candidate feature."
        ),
    )
    parser.add_argument(
        "--n_repeats", type=int, default=2,
        help=(
            "Number of CV repetitions per iteration."
        ),
    )
    parser.add_argument(
        "--test_size", type=float, default=0.25,
        help="Used only when --fit_mode=train_split.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed used by any stochastic components.",
    )
    parser.add_argument(
        "--fit_mode",
        choices=("full", "train_split"),
        default="full",
        help=(
            "Feature-generation fit data. 'full' matches the original "
            "CAAFE usage more closely; 'train_split' preserves the older "
            "benchmark wrapper behavior."
        ),
    )
    parser.add_argument(
        "--datasets_dir", default="./datasets",
        help="Directory containing raw dataset CSVs.",
    )
    parser.add_argument(
        "--output_dir", default="./features",
        help="Root output directory for feature CSVs.",
    )
    parser.add_argument(
        "--method", default=None,
        help=(
            "Output subdirectory name under --output_dir.  "
            "Defaults to 'caafe_<model-slug>' so runs with different LLM "
            "backbones are kept in separate directories for joint evaluation."
        ),
    )
    args = parser.parse_args()
    if args.loop_iterations < 1:
        parser.error("--loop-iterations must be >= 1")

    loader = DatasetLoader()
    _, model = setup_environment()
    method = args.method if args.method is not None else f"caafe_{_model_slug(model)}"
    datasets = _resolve_datasets(args, loader)

    logger.info(
        "Output method directory: %s | run plan: %d dataset(s) x %d loop iteration(s)",
        method,
        len(datasets),
        args.loop_iterations,
    )

    failures: list[str] = []
    for dataset_name in datasets:
        next_run_index = get_next_feature_run_index(args.output_dir, method, dataset_name)
        for offset in range(args.loop_iterations):
            run_index = next_run_index + offset
            run_seed = args.seed + offset
            try:
                _run_single_dataset(
                    args,
                    loader,
                    dataset_name,
                    method,
                    model,
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
