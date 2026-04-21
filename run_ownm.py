"""Run the in-repo ownm feature-generation method and save one feature matrix per run."""

import os
import sys
import logging
import argparse
import time
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_utils import (
    DatasetLoader,
    DatasetValidator,
    FeatureCSVWriter,
    build_feature_output_path,
    get_next_feature_run_index,
)

# Suppress noisy warnings from XGBoost and numpy
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
DEFAULT_API_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

_loader = DatasetLoader()
_validator = DatasetValidator()
_writer = FeatureCSVWriter()


def _usage_attr(obj, attr, default=0):
    """Read usage fields from OpenAI objects or dict-like provider responses."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _response_attr(obj, attr, default=None):
    """Read response fields from OpenAI objects or dict-like provider responses."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _extract_request_id(response):
    """Extract provider request id from common OpenAI-compatible response shapes."""
    request_id = _response_attr(response, "id")
    x_groq = _response_attr(response, "x_groq")
    if x_groq is not None:
        request_id = _response_attr(x_groq, "id", request_id)
    return request_id


def _extract_token_usage(response, iteration: int, elapsed_seconds: float) -> dict:
    """Return a serialisable token-usage record for one API response."""
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
        "iteration": iteration,
        "request_id": _extract_request_id(response),
        "response_created": _response_attr(response, "created"),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def _summarise_token_usage(llm_calls: list[dict]) -> dict:
    """Aggregate per-call usage into the run metadata format."""
    return {
        "llm_call_count": len(llm_calls),
        "input_tokens": sum(call.get("input_tokens", 0) for call in llm_calls),
        "output_tokens": sum(call.get("output_tokens", 0) for call in llm_calls),
        "total_tokens": sum(call.get("total_tokens", 0) for call in llm_calls),
        "calls": llm_calls,
    }


def _generation_summary(outcome_history: list[dict]) -> dict:
    """Summarise accepted/rejected feature batches for run metadata."""
    accepted = [item for item in outcome_history if item["status"] == "accepted"]
    rejected = [item for item in outcome_history if item["status"] == "rejected"]
    return {
        "accepted_batches": len(accepted),
        "rejected_batches": len(rejected),
        "accepted_feature_count": sum(len(item["features"]) for item in accepted),
        "rejected_feature_count": sum(len(item["features"]) for item in rejected),
        "history": outcome_history,
    }


def _chat_completion_options(model: str, temperature: float) -> dict:
    """Build provider options while keeping model-specific tweaks explicit."""
    options = {
        "model": model,
        "temperature": temperature,
    }

    max_tokens = os.environ.get("OWNM_MAX_COMPLETION_TOKENS", "1024").strip()
    if max_tokens:
        options["max_completion_tokens"] = int(max_tokens)

    model_lower = model.lower()
    if "qwen3" in model_lower or model_lower.startswith("qwen/"):
        # Groq supports this for Qwen3. It prevents hidden reasoning from
        # consuming the output budget and truncating the code block.
        options["reasoning_effort"] = os.environ.get("OWNM_REASONING_EFFORT", "none")

    return options


def setup_environment():
    """
    Validate API key and configure endpoint routing.

    Returns
    -------
    str
        Model name to use for LLM calls.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export your API key:\n"
            "  export OPENAI_API_KEY=gsk_your_key_here"
        )

    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_API_BASE_URL)

    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    logger.info("API key: %s...%s", api_key[:8], api_key[-4:])
    logger.info("API base: %s", os.environ["OPENAI_BASE_URL"])
    logger.info("Model: %s", model)
    return model


def import_openai():
    """
    Import the OpenAI client class.

    Returns
    -------
    class
        The OpenAI client class.
    """
    try:
        from openai import OpenAI
        logger.info("Imported OpenAI client (v1+).")
        return OpenAI
    except ImportError as exc:
        raise ImportError(
            "Could not import openai. Install it with:\n"
            '  pip install "openai>=1.0.0"'
        ) from exc


def _strip_code_fences(text):
    """Extract executable Python from LLM output.

    Reasoning models such as Qwen3 may prepend ``<think>...</think>`` blocks
    before a fenced code block. If that wrapper reaches exec(), every
    iteration fails and the run silently saves the unchanged baseline matrix.
    """
    if text is None:
        return ""

    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    fenced_blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_blocks:
        text = "\n\n".join(block.strip() for block in fenced_blocks if block.strip())

    text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _summarise_code_for_prompt(code: str, max_chars: int = 700) -> str:
    """Keep outcome-memory code snippets compact enough for repeated prompts."""
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    snippet = "\n".join(lines)
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def _format_outcome_memory(outcome_history: list[dict], max_items: int = 6) -> str:
    """Format accepted/rejected batches with score deltas and code snippets."""
    if not outcome_history:
        return ""

    accepted = [item for item in outcome_history if item["status"] == "accepted"]
    rejected = [item for item in outcome_history if item["status"] == "rejected"]
    parts = ["Prior feature-generation outcomes:"]

    if accepted:
        parts.append("Accepted batches (improved CV; build on these patterns):")
        for item in accepted[-max_items:]:
            parts.append(
                f"- iter {item['iteration']}: delta={item['delta']:+.5f}, "
                f"features={item['features']}\n"
                f"  code:\n{item['code_summary']}"
            )

    if rejected:
        parts.append("Rejected batches (did not improve CV; avoid similar patterns):")
        for item in rejected[-max_items:]:
            parts.append(
                f"- iter {item['iteration']}: delta={item['delta']:+.5f}, "
                f"features={item['features']}\n"
                f"  code:\n{item['code_summary']}"
            )

    parts.append(
        "Use this feedback to propose genuinely different, executable features. "
        "Do not repeat rejected formulas."
    )
    return "\n" + "\n".join(parts) + "\n"


def _evaluate_features(X_new, y, task_type, cv_folds=3):
    """
    Quick cross-validated score to decide whether to keep new features.

    Uses the same broad objective family as evaluator.py:
    classification uses ROC AUC (binary or OVR macro), regression uses
    negative RMSE. For both metrics, higher is better.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import LabelEncoder

    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        XGBClassifier = GradientBoostingClassifier
        XGBRegressor = GradientBoostingRegressor

    X_numeric = X_new.select_dtypes(include=[np.number])
    if X_numeric.shape[1] == 0:
        raise ValueError("No numeric feature columns available for scoring.")
    X_arr = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0).values

    if task_type == "classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(np.unique(y_enc))
        model = XGBClassifier(n_estimators=50, max_depth=4, random_state=0,
                              eval_metric="logloss")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
        scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
        scores = cross_val_score(model, X_arr, y_enc, cv=cv, scoring=scoring)
        return scores.mean()
    else:
        model = XGBRegressor(n_estimators=50, max_depth=4, random_state=0)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
        scores = cross_val_score(
            model, X_arr, y, cv=cv, scoring="neg_root_mean_squared_error"
        )
        return scores.mean()


def generate_features(OpenAI, X, y, task_type, dataset_name, model, **kwargs):
    """
    Iteratively call an LLM to generate feature-engineering code, execute it,
    and keep features that improve cross-validated performance.

    The loop:
      1. Prompt the LLM with current feature names, dataset context, and
         a natural-language dataset description (from dataset_utils).
      2. Parse and exec() the returned Python code against a copy of X.
      3. Sanitise any inf/NaN values introduced by the generated code.
      4. Evaluate the augmented X with CV; keep if score improves.
      5. Repeat for n_iterations rounds.

    Parameters
    ----------
    OpenAI : class
        The OpenAI client class.
    X : pd.DataFrame
        Original feature matrix.
    y : pd.Series
        Target variable.
    task_type : str
        'classification' or 'regression'.
    dataset_name : str
        Used as context in the LLM prompt.
    model : str
        LLM model name (e.g. 'llama-3.1-8b-instant').
    **kwargs
        n_iterations : int   - LLM rounds (default 10)
        temperature : float  - sampling temperature (default 0.7)
        sleep_between : float - seconds between API calls (default 1.0)
        use_outcomes : bool  - if True, include accepted/rejected feature names
                               in each prompt so the LLM can avoid repeating
                               rejected ideas (default False)
    """
    n_iterations = kwargs.get("n_iterations", 10)
    temperature = kwargs.get("temperature", 0.7)
    sleep_between = kwargs.get("sleep_between", 1.0)
    use_outcomes = kwargs.get("use_outcomes", False)

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
    )

    logger.info("Running ownm: dataset=%s, task=%s, n_iterations=%d, model=%s", dataset_name, task_type, n_iterations, model)
    start_time = time.time()

    X_augmented = X.copy()
    baseline_score = _evaluate_features(X_augmented, y, task_type)
    best_score = baseline_score
    logger.info("Baseline CV score: %.4f", baseline_score)

    outcome_history: list[dict] = []
    llm_calls: list[dict] = []

    # Use dataset description from registry if available
    description = _loader.get_description(dataset_name) or f"Tabular dataset: {dataset_name}"

    system_prompt = (
        "You are a feature engineering expert. Given a tabular dataset, write Python code "
        "that adds new informative features to a DataFrame called X_df. "
        "Use only numpy (imported as np) and existing columns in X_df. "
        "Do not import any other libraries. "
        "Each new feature must be assigned as X_df['descriptive_name'] = ... "
        "Return ONLY executable Python code with no explanation, markdown, "
        "or <think> reasoning tags."
    )

    for i in range(n_iterations):
        numeric_cols = X_augmented.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = X_augmented.columns.tolist()

        outcome_context = _format_outcome_memory(outcome_history) if use_outcomes else ""

        user_prompt = (
            f"Dataset: {dataset_name} ({task_type})\n"
            f"Description: {description}\n"
            f"All columns: {all_cols}\n"
            f"Numeric columns: {numeric_cols}\n"
            f"Shape: {X_augmented.shape[0]} rows x {X_augmented.shape[1]} columns\n"
            f"Current CV score: {best_score:.4f}\n"
            f"{outcome_context}\n"
            "Generate 3-5 new features that may improve predictive performance. "
            "Consider: ratios, products, log transforms, polynomial terms, "
            "binning, and domain-informed combinations."
        )

        try:
            call_start = time.time()
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **_chat_completion_options(model, temperature),
            )
            call_elapsed = time.time() - call_start
            llm_calls.append(_extract_token_usage(response, i + 1, call_elapsed))
            code = _strip_code_fences(response.choices[0].message.content)
            logger.info("Iteration %d/%d: received %d chars of code", i + 1, n_iterations, len(code))
        except Exception as exc:
            logger.warning("Iteration %d: API call failed: %s", i + 1, exc)
            if sleep_between > 0:
                time.sleep(sleep_between * 2)
            continue

        # Execute generated code against a trial copy
        X_trial = X_augmented.copy()
        try:
            exec(code, {"X_df": X_trial, "np": np, "pd": pd})  # noqa: S102
        except Exception as exc:
            logger.warning("Iteration %d: exec() failed: %s", i + 1, exc)
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue

        # Generated code must preserve row alignment exactly. Merges,
        # group-bys, or cartesian expansions can otherwise create a matrix
        # that no longer matches the dataset labels seen by evaluator.py.
        if len(X_trial) != len(X_augmented) or not X_trial.index.equals(X_augmented.index):
            logger.warning(
                "Iteration %d: rejected because generated code changed row alignment "
                "(before=%d rows, after=%d rows).",
                i + 1, len(X_augmented), len(X_trial),
            )
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue

        # Drop any non-numeric columns introduced. Categorical columns need the
        # same treatment as object columns because fillna(0) can fail on a
        # Categorical that does not already include 0 as a category.
        new_cols = [c for c in X_trial.columns if c not in X_augmented.columns]
        for col in new_cols:
            if not pd.api.types.is_numeric_dtype(X_trial[col]):
                try:
                    X_trial[col] = X_trial[col].astype(float)
                except (TypeError, ValueError):
                    X_trial = X_trial.drop(columns=[col])

        new_cols = [c for c in X_trial.columns if c not in X_augmented.columns]
        if not new_cols:
            logger.info("Iteration %d: no usable new columns produced", i + 1)
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue

        # Sanitise numeric inf / NaN values that LLM-generated code may
        # introduce without touching categorical columns.
        numeric_cols_trial = X_trial.select_dtypes(include=[np.number]).columns
        X_trial[numeric_cols_trial] = (
            X_trial[numeric_cols_trial].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        # Remove constant or duplicate generated columns before scoring.
        const_new = [c for c in new_cols if X_trial[c].nunique(dropna=False) <= 1]
        if const_new:
            X_trial = X_trial.drop(columns=const_new)
        new_cols = [c for c in X_trial.columns if c not in X_augmented.columns]

        if new_cols:
            dup_new = X_trial[new_cols].T.duplicated()
            if dup_new.any():
                dup_cols = list(pd.Index(new_cols)[dup_new.to_numpy()])
                X_trial = X_trial.drop(columns=dup_cols)
        new_cols = [c for c in X_trial.columns if c not in X_augmented.columns]

        if not new_cols:
            logger.info("Iteration %d: all generated columns were constant or duplicate", i + 1)
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue

        # Evaluate and keep if improved. Generated code can occasionally
        # create feature matrices that a downstream scorer cannot fit; reject
        # those batches rather than failing the whole dataset run.
        try:
            trial_score = _evaluate_features(X_trial, y, task_type)
        except Exception as exc:
            outcome_history.append(
                {
                    "iteration": i + 1,
                    "status": "failed_scoring",
                    "features": new_cols,
                    "score_before": round(best_score, 6),
                    "score_after": None,
                    "delta": None,
                    "code_summary": _summarise_code_for_prompt(code),
                    "error": str(exc),
                }
            )
            logger.warning("Iteration %d: scoring failed: %s", i + 1, exc)
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue

        delta = trial_score - best_score
        if delta > 0:
            X_augmented = X_trial
            best_score = trial_score
            outcome_history.append(
                {
                    "iteration": i + 1,
                    "status": "accepted",
                    "features": new_cols,
                    "score_before": round(trial_score - delta, 6),
                    "score_after": round(trial_score, 6),
                    "delta": round(delta, 6),
                    "code_summary": _summarise_code_for_prompt(code),
                }
            )
            logger.info("Iteration %d: accepted %d features, score %.4f (+%.4f)", i + 1, len(new_cols), trial_score, delta)
        else:
            outcome_history.append(
                {
                    "iteration": i + 1,
                    "status": "rejected",
                    "features": new_cols,
                    "score_before": round(best_score, 6),
                    "score_after": round(trial_score, 6),
                    "delta": round(delta, 6),
                    "code_summary": _summarise_code_for_prompt(code),
                }
            )
            logger.info("Iteration %d: rejected (score %.4f, delta %.4f)", i + 1, trial_score, delta)

        if sleep_between > 0:
            time.sleep(sleep_between)

    elapsed = time.time() - start_time
    n_orig = X.shape[1]
    n_new = X_augmented.shape[1]
    logger.info("ownm complete in %.1fs: %d -> %d features (%d added), final score: %.4f", elapsed, n_orig, n_new, n_new - n_orig, best_score)
    return X_augmented, _summarise_token_usage(llm_calls), _generation_summary(outcome_history)


def _write_meta(
    output_path: str,
    dataset: str,
    model: str,
    runtime_seconds: float,
    n_original: int,
    n_final: int,
    run_index: int,
    token_usage: dict | None = None,
    generation_outcomes: dict | None = None,
) -> None:
    """Write a JSON sidecar with run metadata alongside the features CSV."""
    import json as _json
    meta = {
        "dataset":          dataset,
        "model":            model,
        "run_index":        run_index,
        "runtime_seconds":  round(runtime_seconds, 2),
        "token_usage":      token_usage or _summarise_token_usage([]),
        "generation_outcomes": generation_outcomes or _generation_summary([]),
        "n_original_features": n_original,
        "n_final_features":    n_final,
        "n_added_features":    n_final - n_original,
    }
    meta_path = output_path.replace("_features.csv", "_meta.json")
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)
    logger.info("Metadata written: %s", meta_path)


def _model_slug(model: str) -> str:
    """
    Convert a model identifier into a filesystem-safe slug.

    Examples
    --------
    "llama-3.3-70b-versatile"        -> "llama-3.3-70b-versatile"
    "openai/gpt-oss-120b"            -> "gpt-oss-120b"
    "moonshotai/kimi-k2-instruct-0905" -> "kimi-k2-instruct-0905"
    "qwen/qwen3-32b"                 -> "qwen3-32b"
    """
    # Strip provider prefix (e.g. "openai/", "qwen/")
    slug = model.split("/")[-1]
    # Replace any remaining characters that are invalid in directory names
    slug = re.sub(r"[^\w\-.]", "_", slug)
    return slug


def _parse_dataset_names(values) -> list[str]:
    """Normalise CLI dataset lists, accepting whitespace or comma-separated names."""
    if not values:
        return []

    names: list[str] = []
    for value in values:
        for part in str(value).split(","):
            name = part.strip()
            if name:
                names.append(name)
    return names


def _resolve_datasets(args) -> list[str]:
    skip_datasets = set(_parse_dataset_names(args.skip_datasets))

    if args.all_datasets:
        datasets = _loader.list_datasets()
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise ValueError("Provide --dataset or use --all-datasets.")

    if skip_datasets:
        datasets = [dataset for dataset in datasets if dataset not in skip_datasets]

    if not datasets:
        raise ValueError("No datasets selected after applying --skip-datasets.")

    unknown_skips = sorted(skip_datasets.difference(_loader.list_datasets()))
    if unknown_skips:
        logger.warning("Ignoring unknown dataset name(s) in --skip-datasets: %s", unknown_skips)

    return datasets


def _resolve_method(args, model: str) -> str:
    if args.method is not None:
        return args.method
    suffix = "_outcomes" if args.use_outcomes else ""
    return f"ownm_{_model_slug(model)}{suffix}"


def _run_single_dataset(
    args,
    dataset_name: str,
    method: str,
    model: str,
    OpenAI,
    run_index: int,
) -> str:
    logger.info(
        "=== ownm runner: dataset=%s | run=%s-%d | n_iterations=%d ===",
        dataset_name,
        dataset_name,
        run_index,
        args.n_iterations,
    )

    X, y, task_type, _feature_names = _loader.load_dataframe(dataset_name, args.datasets_dir)

    report = _validator.validate(X, y, task_type, dataset_name)
    if not report["is_valid"]:
        raise ValueError(f"Dataset validation failed: {report['warnings']}")

    _t0 = time.time()
    X_new, token_usage, generation_outcomes = generate_features(
        OpenAI,
        X,
        y,
        task_type,
        dataset_name,
        model,
        n_iterations=args.n_iterations,
        sleep_between=args.sleep_between,
        use_outcomes=args.use_outcomes,
    )

    output_path = str(
        build_feature_output_path(
            args.output_dir,
            method,
            dataset_name,
            run_index,
        )
    )
    _writer.write(X_new, output_path)
    _write_meta(
        output_path,
        dataset_name,
        model,
        runtime_seconds=time.time() - _t0,
        n_original=X.shape[1],
        n_final=X_new.shape[1],
        run_index=run_index,
        token_usage=token_usage,
        generation_outcomes=generation_outcomes,
    )

    logger.info("=== Done: %s | run=%s-%d -> %s ===", dataset_name, dataset_name, run_index, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run ownm on a dataset."
    )
    parser.add_argument("--dataset", default=None, help="Dataset name (e.g., breast-w, Iris)")
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Run sequentially over every dataset in DatasetLoader.DATASET_METADATA.",
    )
    parser.add_argument(
        "--loop-iterations", type=int, default=1,
        help="Number of full method runs to execute per dataset, saving each run separately.",
    )
    parser.add_argument(
        "--skip-datasets", nargs="*", default=[],
        help=(
            "Dataset name(s) to exclude from this invocation only. Accepts "
            "space-separated values and/or comma-separated lists."
        ),
    )
    parser.add_argument(
        "--method", default=None,
        help=(
            "Output subdirectory name under --output_dir.  "
            "Defaults to 'ownm_<model-slug>' so that runs with different LLM "
            "backbones are automatically kept in separate directories and can "
            "all be evaluated together in a single evaluator.py pass."
        ),
    )
    parser.add_argument(
        "--datasets_dir", default="./datasets", help="Directory containing dataset CSVs"
    )
    parser.add_argument(
        "--output_dir", default="./features", help="Root output directory for feature CSVs"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=10,
        help="Number of LLM feature-generation iterations (default: 10)"
    )
    parser.add_argument(
        "--sleep_between", type=float, default=1.0,
        help="Seconds to sleep between API calls (default: 1.0)"
    )
    parser.add_argument(
        "--use_outcomes", action="store_true",
        help=(
            "If set, each prompt includes the names of previously accepted and "
            "rejected features so the LLM can avoid repeating failed ideas and "
            "build on successful ones.  Ablation flag: compare with and without "
            "to measure the value of outcome memory."
        ),
    )
    args = parser.parse_args()
    if args.loop_iterations < 1:
        parser.error("--loop-iterations must be >= 1")

    model = setup_environment()
    OpenAI = import_openai()
    method = _resolve_method(args, model)
    datasets = _resolve_datasets(args)
    logger.info("Output method directory: %s", method)
    logger.info(
        "Run plan: %d dataset(s) x %d loop iteration(s)",
        len(datasets),
        args.loop_iterations,
    )

    failures: list[str] = []
    for dataset_name in datasets:
        next_run_index = get_next_feature_run_index(args.output_dir, method, dataset_name)
        for offset in range(args.loop_iterations):
            run_index = next_run_index + offset
            try:
                _run_single_dataset(args, dataset_name, method, model, OpenAI, run_index)
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
