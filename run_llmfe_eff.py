"""
LLM-FE Efficiency-Audit Runner
==============================
Thin wrapper around the cloned LLMFE repo (https://github.com/nikhilsab/LLMFE).
The repo's pipeline.main() does all the work - this script just:
  1. Prepares data/ and specs/ in the repo using dataset_utils
  2. Patches the LLM backend to use Groq instead of a local HuggingFace model
  3. Calls pipeline.main() across CV folds (mirroring main.py)
  4. Enforces the paper-style efficiency caps in the upstream sandbox path
  5. Collects the generated column_appender outputs and saves each run's
     features CSV into its own folder

Usage
-----
    python run_llmfe_eff.py --dataset diabetes
    python run_llmfe_eff.py --dataset breast-w --splits 3 --max_samples 10 --loop-iterations 5
    python run_llmfe_eff.py --all-datasets --loop-iterations 3

Python / dependencies
---------------------
In this repo the Groq-backed LLMFE wrapper is run from the shared Python 3.11.7
environment (`benchmark/venvs/venv_llmfe`). The cloned LLMFE repo uses modern
type syntax such as `int | None`, so Python 3.10+ is the safe floor for this
wrapper path.

Environment
-----------
    OPENAI_API_KEY   required
    LLM_MODEL        default: llama-3.3-70b-versatile

Setup
-----
    git clone https://github.com/nikhilsab/LLMFE ./repos/LLMFE
    pip install "openai>=1.0.0" pandas numpy scikit-learn xgboost scipy absl-py requests

Dependency summary for this Groq wrapper:
    - Python 3.10+  (tested here with Python 3.11.7)
    - openai>=1.0.0
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - scipy
    - absl-py
    - requests

The original cloned LLMFE repo ships a broader `requirements.txt` for its
local-model path. This wrapper does not use that HuggingFace inference stack.

This efficiency-audit variant keeps the benchmark-integrated evaluator/output
format, but restores the paper's intended candidate-evaluation constraints more
faithfully by using the upstream timeout setting and an additional memory cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

try:
    from dataset_utils import (
        DatasetLoader,
        FeatureCSVWriter,
        build_feature_output_path,
        get_next_feature_run_index,
    )
except ImportError:  # support `python -m benchmark.run_llmfe`
    from .dataset_utils import (
        DatasetLoader,
        FeatureCSVWriter,
        build_feature_output_path,
        get_next_feature_run_index,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

GROQ_BASE_URL  = "https://api.groq.com/openai/v1"
DEFAULT_MODEL  = "llama-3.3-70b-versatile"


def _humanize_feature_name(name: str) -> str:
    label = name.replace("_", " ").replace("-", " ").strip()
    label = re.sub(r"\s+", " ", label)
    return label


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


def _extract_token_usage(response, prompt_index: int, sample_index: int, elapsed_seconds: float) -> dict:
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
        "prompt_index": prompt_index,
        "sample_index": sample_index,
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


def _require_python_310_plus() -> None:
    """Fail fast with a clear message for unsupported interpreter versions."""
    if sys.version_info < (3, 10):
        version = ".".join(str(x) for x in sys.version_info[:3])
        sys.exit(
            "run_llmfe_eff.py requires Python 3.10+ because the cloned LLMFE repo "
            f"uses modern type syntax (current interpreter: {version})."
        )


@contextmanager
def repo_cwd(repo_path: Path):
    """The repo resolves ./data/, ./logs/, ./specs/ relative to cwd."""
    original = Path.cwd()
    os.chdir(repo_path)
    try:
        yield
    finally:
        os.chdir(original)


# Groq-backed LLM class

def make_groq_llm_class(base_llm_class, api_key: str, model: str, llm_calls: list[dict] | None = None):
    """
    Subclass the repo's LocalLLM, bypassing the HuggingFace/OpenAI backend
    and routing all generation calls to Groq's OpenAI-compatible API.

    Mirrors LocalLLM.__init__ exactly - sets _instruction_prompt, _trim, etc.
    Overrides _draw_samples_api to use the Groq client instead of api.openai.com.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    class GroqLLM(base_llm_class):
        def __init__(self, samples_per_prompt: int, **kwargs):
            # Call grandparent (LLM) __init__ to set _samples_per_prompt
            super(base_llm_class, self).__init__(samples_per_prompt)
            # Set all attributes LocalLLM sets so the pipeline never hits AttributeError
            self._instruction_prompt = (
                "You are a helpful assistant tasked with discovering new features / "
                "dropping less important features for the given prediction task. "
                "Complete the 'modify_features' function below, considering the "
                "physical meaning and relationships of inputs.\n\n"
            )
            self._batch_inference = True
            self._url = "http://127.0.0.1:5000/completions"  # unused but expected
            self._trim = True

        def _draw_samples_api(self, prompt: str, config) -> list:
            """Replace the hardcoded OpenAI call with Groq."""
            from llmfe.sampler import _extract_body
            full_prompt = '\n'.join([self._instruction_prompt, prompt])
            all_samples = []
            prompt_index = len({call.get("prompt_index") for call in (llm_calls or [])}) + 1
            for sample_index in range(1, self._samples_per_prompt + 1):
                for attempt in range(4):
                    try:
                        call_start = time.time()
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.8,
                            max_tokens=2048,
                        )
                        if llm_calls is not None:
                            llm_calls.append(
                                _extract_token_usage(
                                    resp,
                                    prompt_index=prompt_index,
                                    sample_index=sample_index,
                                    elapsed_seconds=time.time() - call_start,
                                )
                            )
                        response = resp.choices[0].message.content
                        if self._trim:
                            response = _extract_body(response, config)
                        all_samples.append(response)
                        break
                    except Exception as exc:
                        wait = 2 ** attempt
                        logger.warning("Groq error (attempt %d/4): %s - retry in %ds",
                                       attempt + 1, exc, wait)
                        time.sleep(wait)
                else:
                    raise RuntimeError("Groq API failed after 4 attempts.")
            return all_samples

    return GroqLLM


# Data and spec preparation

def prepare_repo_data(loader: DatasetLoader, dataset: str, datasets_dir: str, repo_path: Path) -> dict:
    """Write clean CSV + prompt-grounding metadata JSON into <repo>/data/."""
    data_dir = repo_path / "data"
    data_dir.mkdir(exist_ok=True)

    loader.export_clean_csv(dataset, datasets_dir=datasets_dir, dest_dir=str(data_dir))

    meta_path = data_dir / f"{dataset}-metadata.json"
    df = pd.read_csv(data_dir / f"{dataset}.csv")
    target = loader.get_target_column(dataset)
    # Validate the target column actually exists in the exported CSV -
    # dataset_utils may use a normalised name that differs from the raw column name.
    if not target or target not in df.columns:
        logger.warning(
            "Target column '%s' not found in CSV columns - falling back to last column.",
            target
        )
        target = df.columns[-1]
    is_cat = loader.detect_categorical(df, target, dataset_name=dataset)
    feature_metadata = {
        col: _humanize_feature_name(col)
        for col in df.columns
        if col != target
    }
    metadata = {
        "target":              target,
        "is_regression":       loader.get_task_type(dataset) == "regression",
        "label_list":          sorted(df[target].astype(str).unique().tolist()),
        "is_cat":              is_cat,
        "dataset_description": loader.get_description(dataset) or f"Tabular dataset: {dataset}",
        **feature_metadata,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written -> %s", meta_path)
    return metadata


def ensure_spec_file(loader: DatasetLoader, dataset: str, repo_path: Path,
                     feature_names: list[str], target: str, is_regression: bool) -> Path:
    """Write or refresh <repo>/specs/specification_<dataset>.txt for this wrapper."""
    specs_dir = repo_path / "specs"
    specs_dir.mkdir(exist_ok=True)
    spec_path = specs_dir / f"specification_{dataset}.txt"

    description  = loader.get_description(dataset) or f"Tabular dataset: {dataset}"
    task_phrase  = "predicting a continuous value" if is_regression else "predicting the class label"
    feature_list = "\n".join(f"  - {f}" for f in feature_names)

    task_type_str = "regression" if is_regression else "classification"
    metric_str    = "normalized root mean square error (lower is better)" if is_regression \
                    else "accuracy (higher is better)"

    split_line = "X, y, test_size=0.25, random_state=0, stratify=y" if not is_regression \
        else "X, y, test_size=0.25, random_state=0, stratify=None"

    spec_path.write_text(f"""\
\"\"\"
Dataset: {dataset}
{description}

Task: {task_phrase} for target column '{target}'.
Task type: {task_type_str}
Evaluation metric: {metric_str}

Features:
{feature_list}

Generate new informative features that improve predictive performance.
Use domain knowledge where possible. Handle edge cases (zero division,
log of negatives) gracefully. You may also drop redundant features.
\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import preprocessing
import xgboost as xgb


@evaluate.run
def evaluate(data: dict):
    \"\"\"Evaluate the feature transformations on data observations.\"\"\"
    label_encoder = preprocessing.LabelEncoder()
    inputs, outputs = data['inputs'], data['outputs']
    X = modify_features(inputs.copy())
    {"y = label_encoder.fit_transform(np.array(outputs).ravel())" if not is_regression else "y = np.array(outputs, dtype=float).ravel()"}
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == 'string':
            X[col] = label_encoder.fit_transform(X[col].astype(str))
    X = X.fillna(0).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        {split_line})
    {"model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')" if not is_regression else "model = xgb.XGBRegressor(random_state=42)"}
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    {"score = accuracy_score(y_test, y_pred)" if not is_regression else "score = -np.sqrt(mean_squared_error(y_test, y_pred))"}
    return score, inputs, outputs


@equation.evolve
def modify_features(df_input) -> pd.DataFrame:
    \"\"\"Improve the dataset features for {task_type_str}.\"\"\"
    df_output = df_input.copy()
    return df_output
""", encoding="utf-8")
    logger.info("Spec file written -> %s", spec_path)
    return spec_path


def _encode_features_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the final feature matrix to a numeric-only frame for benchmark
    evaluation, while allowing the search itself to operate on raw categoricals.
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


# Feature collection

def collect_features(log_dirs: list[str], X_full: pd.DataFrame, repo_path: Path) -> pd.DataFrame:
    """
    Reads samples_*.json files written by the repo, picks the best-scoring
    function per fold, and applies it to the full dataset.
    """
    import json as _json
    chunks: list[pd.DataFrame] = []
    seen: set[str] = set()

    for fold_i, log_dir in enumerate(log_dirs):
        samples_dir = repo_path / log_dir / "samples"
        if not samples_dir.exists():
            logger.warning("No samples/ in %s - skipping fold.", log_dir)
            continue

        # Load all samples and pick the best scoring one
        best_score = None
        best_func  = None
        for json_file in sorted(samples_dir.glob("samples_*.json")):
            try:
                data = _json.loads(json_file.read_text(encoding="utf-8"))
                score = data.get("score")
                func  = data.get("function", "")
                if score is None or "def modify_features" not in func:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_func  = func
            except Exception as exc:
                logger.debug("Skipping %s: %s", json_file.name, exc)

        if best_func is None:
            logger.warning("No valid scored functions in fold %d - skipping.", fold_i + 1)
            continue

        logger.info("Fold %d best score: %.4f", fold_i + 1, best_score)
        try:
            ns: dict = {}
            exec(best_func, {"pd": pd, "np": np, "preprocessing": preprocessing}, ns)   # noqa: S102
            X_new    = ns["modify_features"](X_full.copy())
            new_cols = [c for c in X_new.columns if c not in X_full.columns]
            if not new_cols:
                logger.info("Fold %d best function added no new columns - skipping.", fold_i + 1)
                continue
            chunk = X_new[new_cols].copy()
            chunk.columns = [f"{c}_f{fold_i}" for c in new_cols]
            chunk = chunk.loc[:, ~chunk.columns.isin(seen)]
            seen.update(chunk.columns)
            chunk = chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
            chunks.append(chunk.reset_index(drop=True))
        except Exception as exc:
            logger.warning("Fold %d: failed to apply best function: %s", fold_i + 1, exc)

    if not chunks:
        logger.warning("No engineered features collected - output will be original features only.")
        return pd.DataFrame(index=range(len(X_full)))

    X_eng = pd.concat(chunks, axis=1)
    # Drop constant / duplicate columns
    X_eng = X_eng.loc[:, X_eng.nunique() > 1]
    X_eng = X_eng.T.drop_duplicates().T
    logger.info("Collected %d engineered feature(s).", X_eng.shape[1])
    return X_eng


# Main

def _write_meta(
    output_path,
    dataset,
    model,
    runtime_seconds,
    n_original,
    n_final,
    run_index,
    seed,
    token_usage=None,
    llmfe_efficiency_config=None,
):
    meta = {
        "dataset":             dataset,
        "model":               model,
        "run_index":           run_index,
        "seed":                seed,
        "runtime_seconds":     round(runtime_seconds, 2),
        "token_usage":         token_usage or _summarise_token_usage([]),
        "n_original_features": n_original,
        "n_final_features":    n_final,
        "n_added_features":    n_final - n_original,
        "llmfe_efficiency_config": llmfe_efficiency_config or {},
    }
    meta_path = output_path.replace("_features.csv", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata written: %s", meta_path)

def _model_slug(model: str) -> str:
    """Convert a model identifier into a filesystem-safe slug, stripping provider prefix."""
    slug = model.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", slug)


def _resolve_datasets(args, loader: DatasetLoader) -> list[str]:
    if args.all_datasets:
        return loader.list_datasets()
    if args.dataset:
        return [args.dataset]
    raise ValueError("Provide --dataset or use --all-datasets.")


def _run_single_dataset(
    args,
    dataset_name: str,
    method: str,
    model: str,
    api_key: str,
    repo_path: Path,
    run_index: int,
    run_seed: int,
) -> str:
    logger.info(
        "=== LLM-FE-EFF | dataset=%s | run=%s-%d | splits=%d | max_samples=%d | timeout=%ds | max_memory_gb=%.2f | seed=%d ===",
        dataset_name,
        dataset_name,
        run_index,
        args.splits,
        args.max_samples,
        args.evaluate_timeout_seconds,
        args.max_memory_gb,
        run_seed,
    )
    _t0 = time.time()

    sys.path.insert(0, str(repo_path))

    from unittest.mock import MagicMock
    for _mod in ("torch", "torch.utils", "torch.utils.tensorboard"):
        sys.modules.setdefault(_mod, MagicMock())

    from llmfe import config as cfg_mod, sampler, evaluator, pipeline
    from utils import is_categorical

    llm_calls: list[dict] = []
    GroqLLM = make_groq_llm_class(sampler.LocalLLM, api_key, model, llm_calls=llm_calls)
    class_config = cfg_mod.ClassConfig(llm_class=GroqLLM, sandbox_class=evaluator.LocalSandbox)
    memory_limit_bytes = None
    if args.max_memory_gb is not None and args.max_memory_gb > 0:
        memory_limit_bytes = int(args.max_memory_gb * (1024 ** 3))
    cfg = cfg_mod.Config(
        use_api=True,
        api_model=model,
        evaluate_timeout_seconds=args.evaluate_timeout_seconds,
        evaluate_max_memory_bytes=memory_limit_bytes,
    )

    loader = DatasetLoader()
    metadata = prepare_repo_data(loader, dataset_name, args.datasets_dir, repo_path)
    is_regression = metadata["is_regression"]
    target = metadata["target"]

    with repo_cwd(repo_path):
        df = pd.read_csv(repo_path / "data" / f"{dataset_name}.csv").convert_dtypes()
        feature_names = [col for col in df.columns if col != target]
        spec_path = ensure_spec_file(loader, dataset_name, repo_path, feature_names, target, is_regression)
        specification = spec_path.read_text(encoding="utf-8")

        X_search = df.drop(columns=[target]).copy().convert_dtypes()
        target_encoder = LabelEncoder()
        y = (
            target_encoder.fit_transform(df[target].astype(str))
            if not is_regression
            else df[target].to_numpy(float)
        )
        is_cat = [is_categorical(X_search.iloc[:, i]) for i in range(X_search.shape[1])]
        logger.info(
            "Prepared LLM-FE search frame: %d rows x %d cols (%d categorical).",
            X_search.shape[0],
            X_search.shape[1],
            int(sum(is_cat)),
        )

        splitter = (
            KFold(n_splits=args.splits, shuffle=True, random_state=run_seed)
            if is_regression
            else StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=run_seed)
        )

        run_tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{run_index}"
        log_dirs: list[str] = []
        for fold_i, (train_idx, _) in enumerate(splitter.split(X_search, y), start=1):
            log_dir = f"logs/{run_tag}_split_{fold_i}"
            log_dirs.append(log_dir)
            logger.info("=== Fold %d/%d ===", fold_i, args.splits)
            try:
                pipeline.main(
                    specification=specification,
                    inputs={"data": {
                        "inputs": X_search.iloc[train_idx],
                        "outputs": y[train_idx],
                        "is_cat": is_cat,
                        "is_regression": is_regression,
                    }},
                    config=cfg,
                    meta_data=metadata,
                    # Upstream sampler divides by 5 internally when checking the cap.
                    max_sample_nums=args.max_samples * 5,
                    class_config=class_config,
                    log_dir=log_dir,
                )
            except Exception as exc:
                logger.error("Fold %d failed: %s", fold_i, exc, exc_info=True)

    X_eng = collect_features(log_dirs, X_search, repo_path)
    X_final = pd.concat([X_search.reset_index(drop=True), X_eng.reset_index(drop=True)], axis=1)
    X_final = _encode_features_for_storage(X_final)

    out_path = str(
        build_feature_output_path(
            args.output_dir,
            method,
            dataset_name,
            run_index,
        )
    )
    FeatureCSVWriter().write(X_final, out_path)
    _write_meta(
        out_path,
        dataset_name,
        model,
        runtime_seconds=time.time() - _t0,
        n_original=X_search.shape[1],
        n_final=X_final.shape[1],
        run_index=run_index,
        seed=run_seed,
        token_usage=_summarise_token_usage(llm_calls),
        llmfe_efficiency_config={
            "evaluate_timeout_seconds": args.evaluate_timeout_seconds,
            "evaluate_max_memory_bytes": memory_limit_bytes,
        },
    )
    logger.info("=== Done | dataset=%s | run=%s-%d | shape=%s | saved -> %s ===", dataset_name, dataset_name, run_index, X_final.shape, out_path)
    return out_path


def main():
    import argparse
    _require_python_310_plus()
    parser = argparse.ArgumentParser(description="Run LLM-FE efficiency audit (Groq/Llama) and save feature CSV.")
    parser.add_argument("--dataset",      default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--loop-iterations", type=int, default=1)
    parser.add_argument("--splits",       type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max_samples",  type=int, default=20)
    parser.add_argument("--evaluate-timeout-seconds", type=int, default=30)
    parser.add_argument("--max-memory-gb", type=float, default=2.0)
    parser.add_argument("--llmfe_dir",    default="./repos/LLMFE")
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("--output_dir",   default="./features")
    parser.add_argument(
        "--method", default=None,
        help=(
            "Output subdirectory name under --output_dir.  "
            "Defaults to 'llmfe_eff_<model-slug>' so efficiency-audit runs are "
            "kept separate from the main benchmark results."
        ),
    )
    args = parser.parse_args()
    if args.loop_iterations < 1:
        parser.error("--loop-iterations must be >= 1")

    api_key = os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY not set.")
    model = os.environ.get("LLM_MODEL") or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    method = args.method if args.method is not None else f"llmfe_eff_{_model_slug(model)}"
    repo_path = Path(args.llmfe_dir).resolve()
    if not repo_path.exists():
        sys.exit(f"LLMFE repo not found at {repo_path}. Clone it first.")

    loader = DatasetLoader()
    datasets = _resolve_datasets(args, loader)
    logger.info(
        "Model: %s | output method directory: %s | run plan: %d dataset(s) x %d loop iteration(s) | timeout=%ds | max_memory_gb=%.2f",
        model,
        method,
        len(datasets),
        args.loop_iterations,
        args.evaluate_timeout_seconds,
        args.max_memory_gb,
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
                    dataset_name,
                    method,
                    model,
                    api_key,
                    repo_path,
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
