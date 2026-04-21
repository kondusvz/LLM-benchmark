"""Run the benchmark-integrated FeatLLM wrapper and save one feature matrix per run."""

from __future__ import annotations

import copy
import hashlib
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam

# dataset_utils must be importable (same directory or on PYTHONPATH)
from dataset_utils import (
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

_DIVIDER = "\n\n---DIVIDER---\n\n"
_VERSION  = "\n\n---VERSION---\n\n"
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


def _extract_token_usage(response, stage: str, prompt_index: int, elapsed_seconds: float) -> dict:
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
        "stage": stage,
        "prompt_index": prompt_index,
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


def _query_gpt_logged(
    text_list,
    api_key,
    llm_calls: list[dict],
    stage: str,
    max_tokens=30,
    temperature=0,
    max_try_num=10,
    model="gpt-3.5-turbo-0613",
):
    """FeatLLM's upstream query_gpt plus exact per-response token logging."""
    import openai
    from tqdm import tqdm

    openai.api_key = api_key
    result_list = []
    for prompt_index, prompt in enumerate(tqdm(text_list), start=1):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                call_start = time.time()
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    request_timeout=100,
                )
                llm_calls.append(
                    _extract_token_usage(
                        response,
                        stage=stage,
                        prompt_index=prompt_index,
                        elapsed_seconds=time.time() - call_start,
                    )
                )
                result = response["choices"][0]["message"]["content"]
                result_list.append(result)
                break
            except openai.error.InvalidRequestError:
                return [-1]
            except Exception as exc:
                print(exc)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list


@contextmanager
def repo_cwd(repo_path: Path):
    """Temporarily set cwd to the FeatLLM repo root."""
    original = Path.cwd()
    os.chdir(repo_path)
    try:
        yield
    finally:
        os.chdir(original)


def setup_environment() -> tuple[str, str]:
    """Validate API key and return (api_key, model_name)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "  export OPENAI_API_KEY=your_key_here"
        )
    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_API_BASE_URL)

    # The FeatLLM repo's utils.query_gpt uses the old openai<1 API.
    # Point it at the right base if using a compatible proxy (e.g. Groq).
    import openai as _openai
    _openai.api_base = os.environ["OPENAI_BASE_URL"]
    _openai.api_key  = api_key

    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    logger.info("API key : %s...%s", api_key[:8], api_key[-4:])
    logger.info("API base: %s", os.environ["OPENAI_BASE_URL"])
    logger.info("Model   : %s", model)
    return api_key, model


def import_utils(featllm_dir: str):
    """Add the FeatLLM repo to sys.path and import utils."""
    repo_path = Path(featllm_dir).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"FeatLLM repo not found at {repo_path}.\n"
            "Clone it with:\n"
            "  git clone https://github.com/Sungwon-Han/FeatLLM ./repos/FeatLLM"
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        import utils as featllm_utils          # noqa: PLC0415
        logger.info("Imported utils from %s", repo_path)
        return featllm_utils, repo_path
    except ImportError as exc:
        raise ImportError(
            f"Could not import utils from {repo_path}. "
            f"Make sure utils.py exists there.\nOriginal error: {exc}"
        ) from exc


def prepare_repo_data(
    loader: DatasetLoader,
    dataset_name: str,
    datasets_dir: str,
    repo_path: Path,
) -> None:
    """
    Ensure the FeatLLM repo has both a cleaned CSV and a metadata JSON
    for this dataset.

    The cleaned CSV is produced by ``DatasetLoader.export_clean_csv``
    (columns dropped, NaN handled, target last).  The raw Kaggle CSV in
    *datasets_dir* is never modified.

    A metadata JSON is generated only when it does not already exist.
    """
    data_dir = repo_path / "data"
    data_dir.mkdir(exist_ok=True)

    # --- Write cleaned CSV (skips if already present) ---
    loader.export_clean_csv(
        dataset_name,
        datasets_dir=datasets_dir,
        dest_dir=str(data_dir),
    )

    # --- Generate metadata JSON ---
    meta_path = data_dir / f"{dataset_name}-metadata.json"
    if meta_path.exists():
        logger.info("Metadata already exists: %s", meta_path)
        return

    logger.info("Generating metadata for '%s' ...", dataset_name)

    csv_dest = data_dir / f"{dataset_name}.csv"
    df_raw = pd.read_csv(csv_dest)

    target_col = loader.get_target_column(dataset_name)
    if target_col is None or target_col not in df_raw.columns:
        target_col = df_raw.columns[-1]

    label_list = sorted(df_raw[target_col].astype(str).unique().tolist())
    is_cat = loader.detect_categorical(df_raw, target_col, dataset_name=dataset_name)

    metadata = {
        "target":     target_col,
        "label_list": label_list,
        "is_cat":     is_cat,
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(
        "Metadata written -> %s  (target='%s', classes=%s)",
        meta_path, target_col, label_list,
    )


# FeatLLM-specific task descriptions, written in the style of the
# original paper's TASK_DICT (natural language, domain-aware, with
# answer format).  Only datasets NOT already in the repo's TASK_DICT
# need an entry here.
_FEATLLM_TASK_DESCRIPTIONS: dict[str, str] = {
    # Labels must exactly match the strings that appear in the target column
    # of the CSV - the LLM will name its functions after these strings, and
    # convert_to_binary matches function names back to labels by substring.
    # Any mismatch causes functions to be silently dropped (label mismatch
    # warning) or assigned to the wrong class (inverted / random AUC).
    #
    # IMPORTANT: labels containing non-identifier characters (<=, >, spaces)
    # must be described with alias names that the LLM will use as valid Python
    # identifiers, AND those aliases must be registered in
    # _LABEL_ALIASES below so convert_to_binary can map them back to the
    # actual label strings in the data.

    # High familiarity
    "Titanic-Dataset":    "Did this passenger survive the Titanic- not_survived (0) or survived (1)-",
    "breast-w":           "Is this breast tissue sample benign or malignant-",
    "diabetes":           "Does this patient have diabetes- not_diabetic (0) or diabetic (1)-",
    "adult":              "Does this person earn more than $50K per year- low_income (<=50K) or high_income (>50K)-",
    "mushroom":           "Is this mushroom edible or poisonous- edible (e) or poisonous (p)-",

    # Medium familiarity
    "bank":               "Did this client subscribe to a term deposit- no or yes-",
    "blood":              "Did this donor donate blood in March 2007- not_donated (0) or donated (1)-",
    "credit-g":           "Is this loan applicant a good or bad credit risk- good or bad-",
    "car":                "How acceptable is this car- unacc (unacceptable), acc (acceptable), good, or vgood (very good)-",
    "spambase":           "Is this email spam- not_spam (0) or spam (1)-",
    "nursery":            "What nursery recommendation applies- not_recom, recommend, very_recom, priority, or spec_prior-",

    # Low familiarity - use exact label strings from the CSV
    "Iris":               "What species is this iris flower- Iris-setosa, Iris-versicolor, or Iris-virginica-",
    "wine":               "How would you rate the quality of this wine- 3, 4, 5, 6, 7, 8, or 9-",
    "ionosphere":         "Does this radar return show a good structure in the ionosphere- good (g) or bad (b)-",
    "MAGIC-gt":           "Is this telescope event a gamma ray or hadron background- gamma (g) or hadron (h)-",

    # german_credit_data: target is good/bad credit risk, NOT loan purpose
    "german_credit_data": "Is this loan applicant a good or bad credit risk- good or bad-",
}

# Maps the alias names used in task descriptions (which the LLM will embed in
# its Python function names) back to the actual label strings in the CSV.
# Only needed when the real label contains characters that cannot appear in a
# Python identifier (<=, >, spaces, etc.).
_LABEL_ALIASES: dict[str, dict[str, str]] = {
    #  dataset          alias           -> real CSV label
    "adult":          {"low_income":    "<=50K",
                       "high_income":   ">50K"},
    "Titanic-Dataset":{"survived":      "1",
                       "not_survived":  "0"},
    "diabetes":       {"diabetic":      "1",
                       "not_diabetic":  "0"},
    "blood":          {"donated":       "1",
                       "not_donated":   "0"},
    "spambase":       {"spam":          "1",
                       "not_spam":      "0"},
    "mushroom":       {"edible":        "e",
                       "poisonous":     "p"},
    "ionosphere":     {"good":          "g",
                       "bad":           "b"},
    "MAGIC-gt":       {"gamma":         "g",
                       "hadron":        "h"},
}


def patch_task_dict(
    utils,
    loader: DatasetLoader,
    dataset_name: str,
    repo_path: Path,
) -> None:
    """
    Add a TASK_DICT entry for this dataset if the repo does not already
    know about it.

    Resolution order:
    1. Already in the repo's TASK_DICT -> do nothing.
    2. In ``_FEATLLM_TASK_DESCRIPTIONS`` -> use that.
    3. Auto-generate from the target column name and class labels.
    """
    if dataset_name in utils.TASK_DICT:
        return

    # Use curated description if available
    if dataset_name in _FEATLLM_TASK_DESCRIPTIONS:
        utils.TASK_DICT[dataset_name] = _FEATLLM_TASK_DESCRIPTIONS[dataset_name]
        logger.info(
            "Patched TASK_DICT for '%s': %s",
            dataset_name, utils.TASK_DICT[dataset_name],
        )
        return

    # Auto-generate fallback using the target column name for context
    csv_path = repo_path / "data" / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)

    target_col = loader.get_target_column(dataset_name)
    if target_col is None or target_col not in df.columns:
        target_col = df.columns[-1]

    classes = sorted(df[target_col].astype(str).unique().tolist())
    # Make the target name more readable: "median_house_value" -> "median house value"
    target_readable = target_col.replace("_", " ").replace("-", " ").lower()

    if len(classes) == 2:
        task_str = (
            f"Based on the given features, predict the {target_readable}. "
            f"Is it '{classes[0]}' or '{classes[1]}'-"
        )
    else:
        class_str = ", ".join(f"'{c}'" for c in classes[:-1]) + f", or '{classes[-1]}'"
        task_str = (
            f"Based on the given features, predict the {target_readable}. "
            f"Is it {class_str}-"
        )

    utils.TASK_DICT[dataset_name] = task_str
    logger.info("Patched TASK_DICT for '%s' (auto-generated): %s", dataset_name, task_str)


def load_dataset(utils, dataset_name: str, shot: int, seed: int, repo_path: Path):
    """
    Call utils.get_dataset from inside the repo cwd so that ./data/ resolves
    correctly.  Returns the full tuple the rest of the pipeline needs.
    """
    logger.info("Loading '%s' via FeatLLM utils (shot=%d, seed=%d)...", dataset_name, shot, seed)
    with repo_cwd(repo_path):
        utils.set_seed(seed)
        (df, X_train, X_test,
         y_train, y_test,
         target_attr, label_list, is_cat) = utils.get_dataset(dataset_name, shot, seed)
    X_all = df.drop(columns=[target_attr])
    logger.info(
        "Dataset loaded: %d rows, %d features, classes=%s",
        len(df), X_all.shape[1], label_list,
    )
    return df, X_all, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat


def _featllm_cache_tag(
    utils,
    dataset_name: str,
    label_list: list,
    num_query: int,
    model: str,
    repo_path: Path,
    cache_namespace: str = "",
) -> str:
    """Build a cache key that changes when prompts or generation settings change."""
    ask_template = (repo_path / "templates" / "ask_llm.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    func_template = (repo_path / "templates" / "ask_for_function.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    payload = {
        "cache_version": 2,
        "dataset": dataset_name,
        "task": utils.TASK_DICT.get(dataset_name, ""),
        "labels": [str(x) for x in label_list],
        "num_query": num_query,
        "model": model,
        "cache_namespace": cache_namespace,
        "ask_template": ask_template,
        "function_template": func_template,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def get_rules(
    utils,
    dataset_name: str,
    X_all, X_train, y_train,
    label_list, target_attr, is_cat,
    api_key: str,
    num_query: int,
    shot: int,
    seed: int,
    repo_path: Path,
    model: str = "gpt-3.5-turbo-0613",
    cache_namespace: str = "",
    llm_calls=None,
) -> tuple[list, object]:
    """Build prompts, call the LLM, and cache results under <repo>/rules/."""
    cache_tag = _featllm_cache_tag(
        utils, dataset_name, label_list, num_query, model, repo_path, cache_namespace
    )
    rule_file = repo_path / "rules" / f"rule-{dataset_name}-{shot}-{seed}-{cache_tag}.out"
    rule_file.parent.mkdir(parents=True, exist_ok=True)

    with repo_cwd(repo_path):
        templates, feature_desc = utils.get_prompt_for_asking(
            dataset_name, X_all, X_train, y_train, label_list,
            target_attr, "./templates/ask_llm.txt",
            f"./data/{dataset_name}-metadata.json",
            is_cat, num_query=num_query,
        )

        if rule_file.exists():
            logger.info("Rule cache found: %s  (skipping LLM call)", rule_file)
            results = rule_file.read_text(encoding='utf-8').strip().split(_DIVIDER)
        else:
            logger.info("Querying LLM for rules (%d prompts)...", len(templates))
            t0 = time.time()
            if llm_calls is None:
                results = utils.query_gpt(
                    templates, api_key, max_tokens=1500, temperature=0.5, model=model
                )
            else:
                results = _query_gpt_logged(
                    templates,
                    api_key,
                    llm_calls=llm_calls,
                    stage="rules",
                    max_tokens=1500,
                    temperature=0.5,
                    model=model,
                )
            logger.info("LLM rule query done in %.1fs", time.time() - t0)
            # Normalise unicode whitespace that cp1252 can't encode (e.g. \u202f)
            results = [r.encode('utf-8', errors='replace').decode('utf-8') for r in results]
            rule_file.write_text(_DIVIDER.join(results), encoding='utf-8')
            logger.info("Rules cached -> %s", rule_file)

    return results, feature_desc


def get_functions(
    utils,
    results,
    feature_desc,
    label_list,
    dataset_name: str,
    api_key: str,
    shot: int,
    seed: int,
    repo_path: Path,
    model: str = "gpt-3.5-turbo-0613",
    cache_namespace: str = "",
    llm_calls=None,
) -> tuple[list, list]:
    """Ask LLM to produce executable Python functions; cache under <repo>/rules/."""
    from tqdm import tqdm

    rules_digest = hashlib.sha1(
        _DIVIDER.join(str(x) for x in results).encode("utf-8", errors="replace")
    ).hexdigest()[:12]
    cache_tag = _featllm_cache_tag(
        utils, dataset_name, label_list, len(results), model, repo_path, cache_namespace
    )
    saved_file = repo_path / "rules" / (
        f"function-{dataset_name}-{shot}-{seed}-{cache_tag}-{rules_digest}.out"
    )
    saved_file.parent.mkdir(parents=True, exist_ok=True)

    with repo_cwd(repo_path):
        parsed_rules = utils.parse_rules(results, label_list)

        if saved_file.exists():
            logger.info("Function cache found: %s  (skipping LLM call)", saved_file)
            total_str   = saved_file.read_text(encoding='utf-8').strip()
            fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]
        else:
            logger.info(
                "Generating Python functions from rules (%d sets)...", len(parsed_rules)
            )
            t0 = time.time()
            fct_strs_all = []
            for parsed_rule in tqdm(parsed_rules, desc="Generating functions"):
                fct_templates = utils.get_prompt_for_generating_function(
                    parsed_rule, feature_desc, "./templates/ask_for_function.txt"
                )
                if llm_calls is None:
                    fct_results = utils.query_gpt(
                        fct_templates, api_key, max_tokens=1500, temperature=0, model=model
                    )
                else:
                    fct_results = _query_gpt_logged(
                        fct_templates,
                        api_key,
                        llm_calls=llm_calls,
                        stage="functions",
                        max_tokens=1500,
                        temperature=0,
                        model=model,
                    )
                fct_strs = []
                for txt in fct_results:
                    if isinstance(txt, int) or "<start>" not in str(txt):
                        logger.warning("  Skipping malformed function response (no <start>)")
                        fct_strs.append("")
                        continue
                    try:
                        fct_strs.append(
                            txt.split("<start>")[1].split("<end>")[0].strip()
                        )
                    except (IndexError, AttributeError):
                        logger.warning("  Skipping malformed function response")
                        fct_strs.append("")
                fct_strs_all.append(fct_strs)
            logger.info("Function generation done in %.1fs", time.time() - t0)
            saved_file.write_text(
                _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all]),
                encoding='utf-8',
            )
            logger.info("Functions cached -> %s", saved_file)

    # Filter sets that could not produce a valid 'def'
    fct_names, fct_strs_final = [], []
    for fct_str_pair in fct_strs_all:
        if not all("def" in s for s in fct_str_pair):
            continue
        try:
            pair_names = [s.split("def")[1].split("(")[0].strip() for s in fct_str_pair]
        except (IndexError, AttributeError):
            continue
        fct_names.append(pair_names)
        fct_strs_final.append(fct_str_pair)

    logger.info("Valid function sets: %d / %d", len(fct_strs_final), len(fct_strs_all))
    return fct_names, fct_strs_final


def convert_to_binary(
    utils, fct_strs_final, fct_names, label_list, X_train, X_test, repo_path: Path,
    dataset_name: str = "", X_all: pd.DataFrame = None,
) -> tuple[list, dict, dict, dict]:
    """Execute LLM-generated functions and produce binary torch tensors.

    Also applies functions to X_all (the full dataset) so the caller can
    save a full-dataset feature matrix.  Saving only the test split causes
    the evaluator to tile the partial matrix, destroying row-label
    correspondence and producing random AUC.
    """
    logger.info("Converting data to per-class binary feature vectors...")
    executable_list: dict  = {}
    X_train_all_dict: dict = {}
    X_test_all_dict: dict  = {}
    X_all_dict: dict       = {}

    # Build a lookup from alias -> real label for this dataset.
    alias_to_label: dict[str, str] = {}
    if dataset_name in _LABEL_ALIASES:
        alias_to_label = _LABEL_ALIASES[dataset_name]
    label_to_alias: dict[str, str] = {v: k for k, v in alias_to_label.items()}

    def _normalise(s: str) -> str:
        import re as _re
        return _re.sub(r'_+', '_', _re.sub(r'[^a-z0-9]', '_', s.lower())).strip('_')

    with repo_cwd(repo_path):
        for i, (fct_str_pair, fct_name_pair) in enumerate(
            zip(fct_strs_final, fct_names)
        ):
            fct_idx_dict: dict = {}
            for idx, name in enumerate(fct_name_pair):
                name_lower = name.lower()
                name_norm  = _normalise(name)
                for label in label_list:
                    if label in fct_idx_dict:
                        continue
                    label_str  = str(label)
                    label_key  = "_".join(label_str.split()).lower()
                    label_norm = _normalise(label_str)

                    if label_key in name_lower:
                        fct_idx_dict[label] = idx
                        continue
                    alias = label_to_alias.get(label_str, "")
                    if alias and alias.lower() in name_lower:
                        fct_idx_dict[label] = idx
                        continue
                    if label_norm and label_norm in name_norm:
                        fct_idx_dict[label] = idx
                        continue
                    for part in label_str.replace("-", " ").split():
                        if len(part) > 2 and part.lower() in name_lower:
                            fct_idx_dict[label] = idx
                            break

            if len(fct_idx_dict) != len(label_list):
                logger.warning(
                    "  Set %d: label mismatch - fct_idx_dict=%s, label_list=%s",
                    i, fct_idx_dict, label_list,
                )
                continue

            try:
                ns = {"pd": pd, "np": np}

                def _clean(df: pd.DataFrame) -> pd.DataFrame:
                    df = df.copy()
                    for col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].astype(str).replace({"nan": "", "<NA>": ""})
                    return df.fillna(0)

                X_train_clean = _clean(X_train)
                X_test_clean  = _clean(X_test)
                X_all_clean   = _clean(X_all) if X_all is not None else None

                X_train_dict: dict = {lbl: {} for lbl in label_list}
                X_test_dict:  dict = {lbl: {} for lbl in label_list}
                X_all_lbl:    dict = {lbl: {} for lbl in label_list}

                for label in label_list:
                    fct_idx = fct_idx_dict[label]
                    exec(fct_strs_final[i][fct_idx].strip('`"'), ns)   # noqa: S102
                    fn = ns[fct_name_pair[fct_idx]]

                    X_tr = fn(X_train_clean).fillna(0).astype("int").to_numpy()
                    X_te = fn(X_test_clean).fillna(0).astype("int").to_numpy()
                    assert X_tr.shape[1] == X_te.shape[1], "Column count mismatch"

                    X_train_dict[label] = torch.tensor(X_tr).float()
                    X_test_dict[label]  = torch.tensor(X_te).float()

                    if X_all_clean is not None:
                        X_a = fn(X_all_clean).fillna(0).astype("int").to_numpy()
                        X_all_lbl[label] = torch.tensor(X_a).float()

                X_train_all_dict[i] = X_train_dict
                X_test_all_dict[i]  = X_test_dict
                X_all_dict[i]       = X_all_lbl
                executable_list[i]  = True

            except Exception as exc:
                logger.warning(
                    "  Set %d: exec failed - %s: %s", i, type(exc).__name__, exc
                )

    exec_keys = list(executable_list.keys())
    logger.info("Executable rule sets: %d", len(exec_keys))
    return exec_keys, X_train_all_dict, X_test_all_dict, X_all_dict






class SimpleModel(nn.Module):
    """Per-class linear scorer (non-negative weights, no bias)."""

    def __init__(self, X_list):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(x.shape[1], 1) / x.shape[1])
            for x in X_list
        ])

    def forward(self, x_list):
        scores = [x @ torch.clamp(w, min=0) for x, w in zip(x_list, self.weights)]
        return torch.cat(scores, dim=-1)


def train_model(X_train_now: list, y_train_num: np.ndarray,
                label_list: list, shot: int) -> SimpleModel:
    """Train a SimpleModel; uses k-fold when shot / n_classes > 1."""
    criterion = nn.CrossEntropyLoss()
    y_tensor  = torch.tensor(y_train_num, dtype=torch.long)

    if shot // len(label_list) == 1:
        model = SimpleModel(X_train_now)
        opt   = Adam(model.parameters(), lr=1e-2)
        for _ in range(200):
            opt.zero_grad()
            outputs = model(X_train_now)
            if (outputs.argmax(dim=1).numpy() == y_train_num).mean() == 1.0:
                break
            criterion(outputs, y_tensor).backward()
            opt.step()
        return model

    n_splits = min(2 if shot // len(label_list) <= 2 else 4, int(np.bincount(y_train_num).min()))
    n_splits = max(n_splits, 2)  # must be at least 2
    if np.bincount(y_train_num).min() < 2:
        # Fall back to no-CV single model if any class has only 1 sample
        model = SimpleModel(X_train_now)
        opt = Adam(model.parameters(), lr=1e-2)
        for _ in range(200):
            opt.zero_grad()
            criterion(model(X_train_now), y_tensor).backward()
            opt.step()
        return model
    kfold    = StratifiedKFold(n_splits=n_splits, shuffle=True)
    models   = []

    for _, (tr_idx, va_idx) in enumerate(kfold.split(X_train_now[0], y_train_num)):
        model = SimpleModel(X_train_now)
        opt   = Adam(model.parameters(), lr=1e-2)
        X_tr  = [x[tr_idx] for x in X_train_now]
        X_va  = [x[va_idx] for x in X_train_now]
        y_tr  = y_train_num[tr_idx]
        y_va  = y_train_num[va_idx]

        best_acc, best_state = -1.0, None
        for _ in range(200):
            opt.zero_grad()
            criterion(model(X_tr), torch.tensor(y_tr, dtype=torch.long)).backward()
            opt.step()
            preds = model(X_va).argmax(dim=1).numpy()
            acc   = (y_va == preds).mean()
            if acc > best_acc:
                best_acc   = acc
                best_state = copy.deepcopy(model.state_dict())
                if best_acc >= 1.0:
                    break
        model.load_state_dict(best_state)
        models.append(model)

    # Average weights across folds
    sdict = models[0].state_dict()
    for key in sdict:
        sdict[key] = torch.stack([m.state_dict()[key] for m in models]).mean(0)
    final = SimpleModel(X_train_now)
    final.load_state_dict(sdict)
    return final


def evaluate_and_collect(
    utils,
    executable_list: list,
    X_train_all_dict: dict,
    X_test_all_dict: dict,
    X_all_dict: dict,
    y_train, y_test,
    label_list: list,
    shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ensemble rule-set models; return full/train/test binary features and AUC.

    The full-dataset binary feature matrix is built directly from X_all_dict,
    which preserves the original dataset row order. Reconstructing the full
    matrix via train+test concatenation is unsafe because the repo split is
    shuffled and therefore does not match the evaluator's label order.
    """
    multiclass  = len(label_list) > 2
    label_list_str = [str(l) for l in label_list]
    auc_labels = list(range(len(label_list_str))) if multiclass else None
    y_train_num = np.array([label_list_str.index(str(k)) for k in y_train])
    y_test_num  = np.array([label_list_str.index(str(k)) for k in y_test])

    test_outputs_all = []
    train_binary_all, test_binary_all, full_binary_all = [], [], []

    for i in executable_list:
        X_tr_now = list(X_train_all_dict[i].values())
        X_te_now = list(X_test_all_dict[i].values())

        trained = train_model(X_tr_now, y_train_num, label_list, shot)
        with torch.no_grad():
            probs = F.softmax(trained(X_te_now), dim=1).numpy()

        auc = utils.evaluate(probs, y_test_num, multiclass=multiclass, labels=auc_labels)
        logger.info("  Rule set %d AUC: %.4f", i, auc)
        test_outputs_all.append(probs)

        # Collect binary features for BOTH splits
        train_binary_all.append(np.concatenate([x.numpy() for x in X_tr_now], axis=1))
        test_binary_all.append( np.concatenate([x.numpy() for x in X_te_now], axis=1))
        full_binary_all.append(
            np.concatenate([x.numpy() for x in X_all_dict[i].values()], axis=1)
        )

    ensembled    = np.stack(test_outputs_all).mean(0)
    ensemble_auc = utils.evaluate(ensembled, y_test_num, multiclass=multiclass, labels=auc_labels)
    logger.info("Ensembled AUC: %.4f", ensemble_auc)

    X_full_features  = np.concatenate(full_binary_all,  axis=1)
    X_train_features = np.concatenate(train_binary_all, axis=1)
    X_test_features  = np.concatenate(test_binary_all,  axis=1)
    logger.info(
        "Binary feature matrix - train: %s, test: %s",
        X_train_features.shape, X_test_features.shape,
    )
    return X_full_features, X_train_features, X_test_features, ensemble_auc


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
        return loader.list_datasets(task_type="classification")
    if args.dataset:
        return [args.dataset]
    raise ValueError("Provide --dataset or use --all-datasets.")


def _run_single_dataset(
    args,
    loader: DatasetLoader,
    utils,
    repo_path: Path,
    dataset_name: str,
    method: str,
    api_key: str,
    model: str,
    run_index: int,
    run_seed: int,
) -> str:
    task_type = loader.get_task_type(dataset_name)
    if task_type == "regression":
        raise ValueError(
            f"Dataset '{dataset_name}' is a regression task. FeatLLM only supports classification."
        )

    logger.info(
        "=== FeatLLM | dataset=%s | run=%s-%d | shot=%d | seed=%d | num_query=%d ===",
        dataset_name,
        dataset_name,
        run_index,
        args.shot,
        run_seed,
        args.num_query,
    )

    prepare_repo_data(loader, dataset_name, args.datasets_dir, repo_path)
    patch_task_dict(utils, loader, dataset_name, repo_path)

    _t0 = time.time()
    _df, X_all, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = (
        load_dataset(utils, dataset_name, args.shot, run_seed, repo_path)
    )

    llm_calls: list[dict] = []
    cache_namespace = f"run-{run_index}"
    results, feature_desc = get_rules(
        utils,
        dataset_name,
        X_all,
        X_train,
        y_train,
        label_list,
        target_attr,
        is_cat,
        api_key,
        args.num_query,
        args.shot,
        run_seed,
        repo_path,
        model=model,
        cache_namespace=cache_namespace,
        llm_calls=llm_calls,
    )

    fct_names, fct_strs_final = get_functions(
        utils,
        results,
        feature_desc,
        label_list,
        dataset_name,
        api_key,
        args.shot,
        run_seed,
        repo_path,
        model=model,
        cache_namespace=cache_namespace,
        llm_calls=llm_calls,
    )

    executable_list, X_train_all_dict, X_test_all_dict, X_all_dict = convert_to_binary(
        utils,
        fct_strs_final,
        fct_names,
        label_list,
        X_train,
        X_test,
        repo_path,
        dataset_name=dataset_name,
        X_all=X_all,
    )

    if not executable_list:
        raise RuntimeError(
            f"No executable rule sets for '{dataset_name}'. "
            "Try --num_query with a higher value or a stronger model."
        )

    X_full_binary, _X_train_bin, _X_test_bin, auc = evaluate_and_collect(
        utils,
        executable_list,
        X_train_all_dict,
        X_test_all_dict,
        X_all_dict,
        y_train,
        y_test,
        label_list,
        args.shot,
    )

    X_orig_full, _y, _task_type, _feature_names = loader.load_dataframe(dataset_name, args.datasets_dir)
    X_features_full = np.concatenate(
        [X_orig_full.to_numpy(dtype=float), X_full_binary],
        axis=1,
    )
    logger.info(
        "Full-dataset feature matrix: %d rows x %d cols (%d original numeric + %d binary)",
        X_features_full.shape[0],
        X_features_full.shape[1],
        X_orig_full.shape[1],
        X_full_binary.shape[1],
    )

    out_path = str(
        build_feature_output_path(
            args.output_dir,
            method,
            dataset_name,
            run_index,
        )
    )
    FeatureCSVWriter().write(X_features_full, out_path)
    _write_meta(
        out_path,
        dataset_name,
        model,
        runtime_seconds=time.time() - _t0,
        n_original=X_all.shape[1],
        n_final=X_features_full.shape[1],
        run_index=run_index,
        seed=run_seed,
        token_usage=_summarise_token_usage(llm_calls),
    )
    logger.info(
        "=== Done | dataset=%s | run=%s-%d | ensemble AUC=%.4f | saved to %s ===",
        dataset_name,
        dataset_name,
        run_index,
        auc,
        out_path,
    )
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run FeatLLM (repo-native) and save binary feature matrix."
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name matching DatasetLoader.DATASET_METADATA (e.g. diabetes, breast-w)",
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Run sequentially over every classification dataset in DatasetLoader.DATASET_METADATA.",
    )
    parser.add_argument(
        "--loop-iterations", type=int, default=1,
        help="Number of full FeatLLM runs to execute per dataset, saving each run separately.",
    )
    parser.add_argument("--shot",       type=int,   default=4,
                        help="Few-shot training examples (default: 4)")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="Random seed for dataset splitting (default: 0)")
    parser.add_argument("--num_query",  type=int,   default=5,
                        help="Number of LLM ensemble queries (default: 5)")
    parser.add_argument("--featllm_dir", default="./repos/FeatLLM",
                        help="Path to cloned FeatLLM repo (default: ./repos/FeatLLM)")
    parser.add_argument("--datasets_dir", default="./datasets",
                        help="Directory containing raw dataset CSVs (default: ./datasets)")
    parser.add_argument("--output_dir",  default="./features",
                        help="Root output directory for feature CSVs (default: ./features)")
    parser.add_argument(
        "--method", default=None,
        help=(
            "Output subdirectory name under --output_dir.  "
            "Defaults to 'featllm_<model-slug>' so runs with different LLM "
            "backbones are kept in separate directories for joint evaluation."
        ),
    )
    args = parser.parse_args()
    if args.loop_iterations < 1:
        parser.error("--loop-iterations must be >= 1")

    loader = DatasetLoader()
    api_key, model = setup_environment()
    method = args.method if args.method is not None else f"featllm_{_model_slug(model)}"
    utils, repo_path = import_utils(args.featllm_dir)
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
                    utils,
                    repo_path,
                    dataset_name,
                    method,
                    api_key,
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
