"""Benchmark evaluation utilities.

Evaluates saved feature runs across benchmark datasets, compares them against
baseline features, and writes raw and aggregated reports.
"""

import os
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from dataset_utils import DatasetLoader

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_WIDTH = 96

METHOD_DISPLAY_NAMES = {
    "caafe_gpt-oss-120b": "CAAFE (GPT-OSS-120B)",
    "caafe_llama-3.3-70b-versatile": "CAAFE (Llama 3.3 70B)",
    "featllm_gpt-oss-120b": "FeatLLM (GPT-OSS-120B)",
    "featllm_llama-3.3-70b-versatile": "FeatLLM (Llama 3.3 70B)",
    "llmfe_gpt-oss-120b-pre-logging": "LLM-FE (GPT-OSS-120B)",
    "llmfe_llama-3.3-70b-versatile": "LLM-FE (Llama 3.3 70B)",
    "openfe": "OpenFE",
    "ownm_gpt-oss-120b": "ownm (GPT-OSS-120B)",
    "ownm_llama-3.3-70b-versatile": "ownm (Llama 3.3 70B)",
    "ownm_qwen3-32b": "ownm (Qwen3 32B)",
}


def _display_method_name(method: str) -> str:
    """Return a report-friendly label for a saved method directory name."""
    if method in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[method]
    parts = str(method).replace("_", " ").split()
    return " ".join(part.upper() if part.isupper() else part.capitalize() for part in parts)


def _sort_methods(methods: List[str]) -> List[str]:
    """Sort methods by display name for stable, readable report tables."""
    return sorted(methods, key=lambda method: _display_method_name(method).lower())


class DatasetProfiler:
    """
    Provides static metadata for the supported benchmark datasets and runtime
    classification of dataset characteristics across four axes:
    sample size, dimensionality, feature type, and LLM familiarity.

    Static metadata covers:
    - task_type: "classification" or "regression"
    - llm_familiarity: "High", "Medium", or "Low"
    - n_features: approximate feature count (from schema knowledge)

    Runtime axes are computed from the actual loaded data to reflect the
    true characteristics of each dataset after preprocessing.
    """

    _LOADER = DatasetLoader()

    @staticmethod
    def get_metadata(dataset_name: str) -> dict:
        """
        Return static metadata for a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (must match a key in _METADATA).

        Returns
        -------
        dict
            Keys: task_type, llm_familiarity, n_features.
            Returns a dict with None values if the dataset is unknown.
        """
        meta = DatasetProfiler._LOADER.DATASET_METADATA.get(dataset_name)
        if meta is None:
            return {"task_type": None, "llm_familiarity": "Unknown", "n_features": None}
        task_type, _target_col, n_features = meta
        return {
            "task_type": task_type,
            "llm_familiarity": DatasetProfiler._LOADER.get_llm_familiarity(dataset_name),
            "n_features": n_features,
        }

    @staticmethod
    def classify_sample_size(n_rows: int) -> str:
        """
        Classify a dataset by number of rows.

        Parameters
        ----------
        n_rows : int
            Number of samples in the dataset.

        Returns
        -------
        str
            "Small" (< 500), "Medium" (500-5000), or "Large" (> 5000).
        """
        if n_rows < 500:
            return "Small"
        if n_rows <= 5000:
            return "Medium"
        return "Large"

    @staticmethod
    def classify_dimensionality(n_features: int) -> str:
        """
        Classify a dataset by number of features.

        Parameters
        ----------
        n_features : int
            Number of feature columns.

        Returns
        -------
        str
            "Low" (< 10), "Medium" (10-30), or "High" (> 30).
        """
        if n_features < 10:
            return "Low"
        if n_features <= 30:
            return "Medium"
        return "High"

    @staticmethod
    def classify_feature_type(
        loader: DatasetLoader,
        dataset_name: str,
        X_df: pd.DataFrame,
    ) -> str:
        """
        Classify the predominant feature type in a DataFrame.

        Parameters
        ----------
        loader : DatasetLoader
            Shared loader used for consistent categorical detection.
        dataset_name : str
            Dataset name so per-dataset categorical overrides can be applied.
        X_df : pd.DataFrame
            Feature matrix with original dtypes.

        Returns
        -------
        str
            "Numerical" if < 20% of columns are categorical,
            "Categorical" if > 80% of columns are categorical,
            "Mixed" otherwise.
        """
        if X_df.shape[1] == 0:
            return "Numerical"
        df_profile = X_df.copy()
        df_profile["__target__"] = 0
        is_cat = loader.detect_categorical(
            df_profile,
            "__target__",
            dataset_name=dataset_name,
        )
        cat_ratio = sum(is_cat.values()) / len(is_cat)
        if cat_ratio < 0.20:
            return "Numerical"
        if cat_ratio > 0.80:
            return "Categorical"
        return "Mixed"


class UniversalEvaluator:
    """
    Trains an XGBoost model across multiple random seeds and reports
    averaged metrics.

    For classification, reports accuracy and ROC AUC.
    For regression, reports Normalised RMSE (NRMSE).

    Parameters
    ----------
    task_type : str
        "classification" or "regression".
    n_seeds : int, optional
        Number of random seeds to average over (default: 5).
    test_size : float, optional
        Fraction of data held out for testing (default: 0.2).
    """

    def __init__(self, task_type: str, n_seeds: int = 5, test_size: float = 0.2):
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")
        self.task_type = task_type
        self.n_seeds = n_seeds
        self.test_size = test_size

    def evaluate(
        self,
        X_orig: np.ndarray,
        X_feat: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Train XGBoost on both X_orig (baseline) and X_feat (generated),
        evaluate on held-out test splits.

        Parameters
        ----------
        X_orig : np.ndarray
            Original feature matrix (used for baseline scores).
        X_feat : np.ndarray
            Generated / engineered feature matrix for training.
        y : np.ndarray
            Target array.

        Returns
        -------
        dict
            Keys:
            - mean_accuracy, std_accuracy  (None for regression)
            - mean_auc, std_auc            (None for regression)
            - mean_nrmse, std_nrmse        (None for classification)
            - baseline_accuracy_mean, baseline_accuracy_std
            - baseline_auc_mean, baseline_auc_std
            - baseline_nrmse_mean, baseline_nrmse_std
            - n_seeds                      (int)
            - feature_count                (int, columns in X_feat)
        """
        accuracies: List[float] = []
        aucs: List[float] = []
        nrmses: List[float] = []
        bl_accuracies: List[float] = []
        bl_aucs: List[float] = []
        bl_nrmses: List[float] = []

        for seed in range(self.n_seeds):
            # Generated features
            acc, auc, nrmse = self._train_and_score(X_feat, y, seed)
            if acc is not None:
                accuracies.append(acc)
            if auc is not None:
                aucs.append(auc)
            if nrmse is not None:
                nrmses.append(nrmse)

            # Baseline (original features)
            bl_acc, bl_auc, bl_nrmse = self._train_and_score(X_orig, y, seed)
            if bl_acc is not None:
                bl_accuracies.append(bl_acc)
            if bl_auc is not None:
                bl_aucs.append(bl_auc)
            if bl_nrmse is not None:
                bl_nrmses.append(bl_nrmse)

        result = {
            "mean_accuracy":  float(np.mean(accuracies))  if accuracies else None,
            "std_accuracy":   float(np.std(accuracies))   if accuracies else None,
            "mean_auc":       float(np.mean(aucs))        if aucs       else None,
            "std_auc":        float(np.std(aucs))         if aucs       else None,
            "mean_nrmse":     float(np.mean(nrmses))      if nrmses     else None,
            "std_nrmse":      float(np.std(nrmses))       if nrmses     else None,
            "baseline_accuracy_mean": float(np.mean(bl_accuracies)) if bl_accuracies else None,
            "baseline_accuracy_std":  float(np.std(bl_accuracies))  if bl_accuracies else None,
            "baseline_auc_mean":      float(np.mean(bl_aucs))       if bl_aucs       else None,
            "baseline_auc_std":       float(np.std(bl_aucs))        if bl_aucs       else None,
            "baseline_nrmse_mean":    float(np.mean(bl_nrmses))     if bl_nrmses     else None,
            "baseline_nrmse_std":     float(np.std(bl_nrmses))      if bl_nrmses     else None,
            "n_seeds":        self.n_seeds,
            "feature_count":  X_feat.shape[1] if X_feat.ndim == 2 else 1,
        }
        return result

    def _build_model(self, seed: int):
        """Instantiate an XGBoost model with fixed hyper-parameters."""
        common_kwargs = {
            "n_estimators":  100,
            "max_depth":     6,
            "learning_rate": 0.1,
            "random_state":  seed,
        }
        if self.task_type == "classification":
            return XGBClassifier(
                **common_kwargs,
                eval_metric="logloss",
            )
        return XGBRegressor(**common_kwargs)

    def _train_and_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Split data, train model, and compute metrics for a single seed.

        Returns
        -------
        Tuple[Optional[float], Optional[float], Optional[float]]
            (accuracy, auc, nrmse) - unused metrics are None.
        """
        if self.task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=seed,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=seed,
            )

        model = self._build_model(seed)
        model.fit(X_train, y_train)

        if self.task_type == "classification":
            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))

            # ROC AUC - handle binary and multi-class
            try:
                n_classes = len(np.unique(y))
                if n_classes == 2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc = float(roc_auc_score(y_test, y_prob))
                else:
                    y_prob = model.predict_proba(X_test)
                    try:
                        auc = float(
                            roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                        )
                    except ValueError as exc:
                        present_classes = np.unique(y_test)
                        if len(present_classes) >= 2 and len(present_classes) < y_prob.shape[1]:
                            y_prob_present = y_prob[:, present_classes]
                            row_sums = y_prob_present.sum(axis=1, keepdims=True)
                            row_sums[row_sums == 0] = 1.0
                            y_prob_present = y_prob_present / row_sums
                            auc = float(
                                roc_auc_score(
                                    y_test,
                                    y_prob_present,
                                    labels=present_classes,
                                    multi_class="ovr",
                                    average="macro",
                                )
                            )
                            logger.debug(
                                "ROC AUC fallback used for seed %d: test fold contained %d/%d classes.",
                                seed,
                                len(present_classes),
                                y_prob.shape[1],
                            )
                        else:
                            raise exc
            except Exception as exc:
                logger.debug("ROC AUC computation failed for seed %d: %s", seed, exc)
                auc = None

            return acc, auc, None

        else:  # regression
            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            y_range = float(y_test.max() - y_test.min())
            if y_range == 0.0:
                y_range = float(np.std(y_test)) or 1.0  # fallback to std
            nrmse = rmse / y_range
            return None, None, nrmse


class BenchmarkEvaluator:
    """
    Main orchestrator for the LLM feature engineering benchmark.

    Scans the features directory for saved feature sets, loads the
    corresponding original datasets, runs UniversalEvaluator, and
    produces both per-feature-set results and aggregated mean reports.

    Parameters
    ----------
    input_dir : str
        Root directory containing method sub-folders with feature CSVs.
        Supported layouts:
        - {input_dir}/{method}/{dataset}_features.csv
        - {input_dir}/{method}/{dataset}/{dataset}-{run}/{dataset}_features.csv
    output_dir : str
        Directory where results.csv and the benchmark report are written.
    datasets_dir : str
        Directory containing original dataset CSVs: {datasets_dir}/{dataset}.csv
    test_size : float, optional
        Fraction held out for testing (default: 0.2).
    n_seeds : int, optional
        Number of random seeds per evaluation (default: 5).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        datasets_dir: str,
        test_size: float = 0.2,
        n_seeds: int = 5,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.datasets_dir = datasets_dir
        self.test_size = test_size
        self.n_seeds = n_seeds
        self._profiler = DatasetProfiler()
        self._loader = DatasetLoader()

    def discover_feature_sets(self) -> List[dict]:
        """
        Scan input_dir for saved feature sets.

        Supported layouts:
        - Legacy: {input_dir}/{method}/{dataset}_features.csv
        - Multi-run: {input_dir}/{method}/{dataset}/{dataset}-{run}/{dataset}_features.csv

        Returns
        -------
        list of dict
            Sorted feature-set records with keys:
            method, dataset, feature_set, run_index, feature_path.
        """
        feature_sets: List[dict] = []
        valid_datasets = set(self._loader.list_datasets())
        if not os.path.isdir(self.input_dir):
            logger.warning("input_dir does not exist: %s", self.input_dir)
            return feature_sets

        for method in sorted(os.listdir(self.input_dir)):
            method_dir = os.path.join(self.input_dir, method)
            if not os.path.isdir(method_dir):
                continue

            for root, _dirs, files in os.walk(method_dir):
                for fname in sorted(files):
                    if not fname.endswith("_features.csv"):
                        continue

                    feature_path = os.path.join(root, fname)
                    rel_parts = os.path.relpath(feature_path, method_dir).split(os.sep)

                    if len(rel_parts) == 1:
                        dataset = fname[: -len("_features.csv")]
                        feature_set = "legacy"
                        run_index = 0
                    else:
                        dataset = rel_parts[0]
                        feature_set = rel_parts[-2]
                        prefix = f"{dataset}-"
                        if feature_set.startswith(prefix):
                            suffix = feature_set[len(prefix):]
                            run_index = int(suffix) if suffix.isdigit() else None
                        else:
                            run_index = None

                    if dataset not in valid_datasets:
                        logger.info(
                            "Skipping feature set for retired or unknown dataset '%s': %s",
                            dataset,
                            feature_path,
                        )
                        continue

                    feature_sets.append({
                        "method": method,
                        "dataset": dataset,
                        "feature_set": feature_set,
                        "run_index": run_index,
                        "feature_path": feature_path,
                    })

        feature_sets.sort(
            key=lambda item: (
                item["method"],
                item["dataset"],
                item["run_index"] if item["run_index"] is not None else float("inf"),
                item["feature_set"],
                item["feature_path"],
            )
        )
        logger.info("Discovered %d feature sets.", len(feature_sets))
        return feature_sets

    def load_dataset(
        self, dataset_name: str
    ) -> Tuple[pd.DataFrame, pd.Series, str, pd.DataFrame]:
        """
        Load an original dataset through DatasetLoader so evaluation uses the
        same cleaning, target resolution, and encoding as the runners.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (without .csv extension).

        Returns
        -------
        tuple
            (X_df, y_series, task_type, X_raw_df)
            X_df is numeric/encoded for modelling; X_raw_df preserves cleaned
            pre-encoding dtypes for feature-type profiling.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        """
        X_df, y, task_type, _ = self._loader.load_dataframe(dataset_name, self.datasets_dir)
        df_raw, target_col, _ = self._loader.read_and_clean(dataset_name, self.datasets_dir)
        X_raw_df = df_raw.drop(columns=[target_col]).copy()
        return X_df, y, task_type, X_raw_df

    def load_features(self, feature_path: str) -> Optional[pd.DataFrame]:
        """
        Load a pre-generated features CSV.

        Parameters
        ----------
        feature_path : str
            Full path to a pre-generated feature CSV.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of features, or None if the file does not exist.
        """
        if not os.path.isfile(feature_path):
            logger.warning("Features file not found: %s", feature_path)
            return None

        feat_df = pd.read_csv(feature_path)

        # Feature rows must correspond 1:1 with the original dataset rows.
        # Partial matrices are rejected in run_evaluation() so row/label
        # alignment bugs do not get silently masked.
        return feat_df

    def run_evaluation(self) -> pd.DataFrame:
        """
        Execute the full evaluation loop over all discovered feature sets.

        For each feature set:
        - Loads the original dataset (once per unique dataset name, cached).
        - Loads the corresponding features CSV.
        - Runs UniversalEvaluator with the configured seeds and test size.
        - Records results with axis metadata.

        Returns
        -------
        pd.DataFrame
            One row per saved feature set with columns:
            method, dataset, feature_set, run_index, feature_path,
            task_type, llm_familiarity,
            sample_size_axis, dimensionality_axis, feature_type_axis,
            n_samples, n_original_features, n_generated_features,
            accuracy_mean, accuracy_std, auc_mean, auc_std,
            nrmse_mean, nrmse_std, status, error_message.
        """
        feature_sets = self.discover_feature_sets()
        dataset_cache: Dict[str, Tuple[pd.DataFrame, pd.Series, str, pd.DataFrame]] = {}
        rows: List[dict] = []

        for feature_set in feature_sets:
            method = feature_set["method"]
            dataset_name = feature_set["dataset"]
            feature_set_name = feature_set["feature_set"]
            run_index = feature_set["run_index"]
            feature_path = feature_set["feature_path"]
            logger.info(
                "Evaluating method='%s' dataset='%s' feature_set='%s'",
                method,
                dataset_name,
                feature_set_name,
            )
            row: dict = {
                "method":               method,
                "dataset":              dataset_name,
                "feature_set":          feature_set_name,
                "run_index":            run_index,
                "feature_path":         feature_path,
                "task_type":            None,
                "llm_familiarity":      None,
                "sample_size_axis":     None,
                "dimensionality_axis":  None,
                "feature_type_axis":    None,
                "n_samples":            None,
                "n_original_features":  None,
                "n_generated_features": None,
                "accuracy_mean":        None,
                "accuracy_std":         None,
                "auc_mean":             None,
                "auc_std":              None,
                "nrmse_mean":           None,
                "nrmse_std":            None,
                "baseline_accuracy_mean": None,
                "baseline_accuracy_std":  None,
                "baseline_auc_mean":      None,
                "baseline_auc_std":       None,
                "baseline_nrmse_mean":    None,
                "baseline_nrmse_std":     None,
                "status":               "pending",
                "error_message":        None,
            }

            feat_df = self.load_features(feature_path)
            if feat_df is None:
                row["status"] = "missing"
                rows.append(row)
                continue

            try:
                if dataset_name not in dataset_cache:
                    dataset_cache[dataset_name] = self.load_dataset(dataset_name)
                X_orig_df, y, task_type, X_profile_df = dataset_cache[dataset_name]
            except Exception as exc:
                logger.error(
                    "Failed to load dataset '%s': %s", dataset_name, exc
                )
                row["status"] = "failed"
                row["error_message"] = f"Dataset load error: {exc}"
                rows.append(row)
                continue

            meta = DatasetProfiler.get_metadata(dataset_name)
            row["task_type"] = task_type
            row["llm_familiarity"] = meta.get("llm_familiarity", "Unknown")
            row["n_samples"] = len(y)
            row["n_original_features"] = X_orig_df.shape[1]
            row["n_generated_features"] = feat_df.shape[1]
            row["sample_size_axis"] = DatasetProfiler.classify_sample_size(len(y))
            row["dimensionality_axis"] = DatasetProfiler.classify_dimensionality(
                X_orig_df.shape[1]
            )
            row["feature_type_axis"] = DatasetProfiler.classify_feature_type(
                self._loader,
                dataset_name,
                X_profile_df,
            )

            # Feature rows must align 1:1 with the original dataset rows.
            # Tiling/truncating hides runner bugs and destroys the
            # feature-label correspondence needed for meaningful AUC.
            n_full = len(y)
            n_feat = len(feat_df)
            if n_feat != n_full:
                msg = (
                    f"Feature matrix row mismatch: got {n_feat} rows, "
                    f"expected {n_full}. Generated features must be saved "
                    f"for the full dataset in the same row order."
                )
                logger.error(
                    "method='%s' dataset='%s' feature_set='%s': %s",
                    method, dataset_name, feature_set_name, msg,
                )
                row["status"] = "failed"
                row["error_message"] = msg
                rows.append(row)
                continue

            try:
                X_orig_np = X_orig_df.values.astype(float)
                X_feat_np = feat_df.values.astype(float)
                y_np = y.values

                evaluator = UniversalEvaluator(
                    task_type=task_type,
                    n_seeds=self.n_seeds,
                    test_size=self.test_size,
                )
                results = evaluator.evaluate(X_orig_np, X_feat_np, y_np)

                row["accuracy_mean"] = results["mean_accuracy"]
                row["accuracy_std"]  = results["std_accuracy"]
                row["auc_mean"]      = results["mean_auc"]
                row["auc_std"]       = results["std_auc"]
                row["nrmse_mean"]    = results["mean_nrmse"]
                row["nrmse_std"]     = results["std_nrmse"]
                row["baseline_accuracy_mean"] = results["baseline_accuracy_mean"]
                row["baseline_accuracy_std"]  = results["baseline_accuracy_std"]
                row["baseline_auc_mean"]      = results["baseline_auc_mean"]
                row["baseline_auc_std"]       = results["baseline_auc_std"]
                row["baseline_nrmse_mean"]    = results["baseline_nrmse_mean"]
                row["baseline_nrmse_std"]     = results["baseline_nrmse_std"]
                row["status"] = "success"

            except Exception as exc:
                logger.error(
                    "Evaluation failed for method='%s' dataset='%s' feature_set='%s': %s",
                    method, dataset_name, feature_set_name, exc,
                )
                row["status"] = "failed"
                row["error_message"] = str(exc)

            rows.append(row)

        results_df = pd.DataFrame(rows)
        logger.info(
            "Evaluation complete. %d total feature sets, %d succeeded, %d failed, %d missing.",
            len(results_df),
            (results_df["status"] == "success").sum(),
            (results_df["status"] == "failed").sum(),
            (results_df["status"] == "missing").sum(),
        )
        return results_df

    def aggregate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-feature-set results into one mean row per method/dataset.

        Metrics are averaged across successful feature sets. Standard-deviation
        columns in the aggregated table reflect variation across feature-set
        means, which is the quantity the user requested to compare.
        """
        if results_df.empty:
            return results_df.copy()

        metric_cols = [
            "accuracy_mean",
            "auc_mean",
            "nrmse_mean",
            "baseline_accuracy_mean",
            "baseline_auc_mean",
            "baseline_nrmse_mean",
        ]
        first_cols = [
            "task_type",
            "llm_familiarity",
            "sample_size_axis",
            "dimensionality_axis",
            "feature_type_axis",
            "n_samples",
            "n_original_features",
        ]
        aggregated_rows: List[dict] = []

        for (method, dataset_name), group in results_df.groupby(["method", "dataset"], sort=True):
            success_group = group[group["status"] == "success"].copy()
            source = success_group.iloc[0] if not success_group.empty else group.iloc[0]

            row = {
                "method": method,
                "dataset": dataset_name,
                "feature_set_count": int(len(group)),
                "successful_feature_sets": int((group["status"] == "success").sum()),
                "failed_feature_sets": int((group["status"] == "failed").sum()),
                "missing_feature_sets": int((group["status"] == "missing").sum()),
            }
            for col in first_cols:
                row[col] = source.get(col)

            if success_group.empty:
                row["n_generated_features"] = None
                row["accuracy_mean"] = None
                row["accuracy_std"] = None
                row["auc_mean"] = None
                row["auc_std"] = None
                row["nrmse_mean"] = None
                row["nrmse_std"] = None
                row["baseline_accuracy_mean"] = None
                row["baseline_accuracy_std"] = None
                row["baseline_auc_mean"] = None
                row["baseline_auc_std"] = None
                row["baseline_nrmse_mean"] = None
                row["baseline_nrmse_std"] = None
                row["status"] = source.get("status", "failed")
                row["error_message"] = "; ".join(
                    msg for msg in group["error_message"].dropna().astype(str).unique()
                ) or None
                aggregated_rows.append(row)
                continue

            row["n_generated_features"] = float(success_group["n_generated_features"].mean())
            row["accuracy_mean"] = float(success_group["accuracy_mean"].mean()) if success_group["accuracy_mean"].notna().any() else None
            row["accuracy_std"] = float(success_group["accuracy_mean"].std(ddof=0)) if success_group["accuracy_mean"].notna().any() else None
            row["auc_mean"] = float(success_group["auc_mean"].mean()) if success_group["auc_mean"].notna().any() else None
            row["auc_std"] = float(success_group["auc_mean"].std(ddof=0)) if success_group["auc_mean"].notna().any() else None
            row["nrmse_mean"] = float(success_group["nrmse_mean"].mean()) if success_group["nrmse_mean"].notna().any() else None
            row["nrmse_std"] = float(success_group["nrmse_mean"].std(ddof=0)) if success_group["nrmse_mean"].notna().any() else None
            row["baseline_accuracy_mean"] = float(success_group["baseline_accuracy_mean"].mean()) if success_group["baseline_accuracy_mean"].notna().any() else None
            row["baseline_accuracy_std"] = float(success_group["baseline_accuracy_mean"].std(ddof=0)) if success_group["baseline_accuracy_mean"].notna().any() else None
            row["baseline_auc_mean"] = float(success_group["baseline_auc_mean"].mean()) if success_group["baseline_auc_mean"].notna().any() else None
            row["baseline_auc_std"] = float(success_group["baseline_auc_mean"].std(ddof=0)) if success_group["baseline_auc_mean"].notna().any() else None
            row["baseline_nrmse_mean"] = float(success_group["baseline_nrmse_mean"].mean()) if success_group["baseline_nrmse_mean"].notna().any() else None
            row["baseline_nrmse_std"] = float(success_group["baseline_nrmse_mean"].std(ddof=0)) if success_group["baseline_nrmse_mean"].notna().any() else None
            row["status"] = "success"
            row["error_message"] = None
            aggregated_rows.append(row)

        return pd.DataFrame(aggregated_rows)

    def generate_report(self, results_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
        """
        Build a formatted plaintext benchmark report.

        Report sections:
        1. Header with timestamp and summary statistics.
        2. Overall Method Rankings (classification by AUC, regression by NRMSE).
        3. Per-dataset accuracy table (pivot: dataset x method).
        4. Per-dataset AUC table (classification only).
        5. Axis-stratified analysis for Sample Size, Dimensionality,
           Feature Type, and LLM Familiarity axes.
        6. Footer with methodology notes.

        Parameters
        ----------
        results_df : pd.DataFrame
            Raw per-feature-set output from run_evaluation().
        summary_df : pd.DataFrame
            Aggregated one-row-per-method/dataset summary from aggregate_results().

        Returns
        -------
        str
            Formatted report text.
        """
        lines: List[str] = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success_df = summary_df[summary_df["status"] == "success"].copy()
        clf_df = success_df[success_df["task_type"] == "classification"]
        reg_df = success_df[success_df["task_type"] == "regression"]

        lines.extend(self._section_header("Benchmark Evaluation Report"))
        lines.append("Feature-generation methods compared against an XGBoost baseline.")
        lines.append(f"Prepared: {ts}")
        lines.append("")

        lines.extend(self._section_header("Summary", char="-"))
        lines.append(self._kv_line("Feature-set evaluations", len(results_df)))
        lines.append(self._kv_line("Successful feature sets", int((results_df["status"] == "success").sum())))
        lines.append(self._kv_line("Failed feature sets", int((results_df["status"] == "failed").sum())))
        lines.append(self._kv_line("Missing feature files", int((results_df["status"] == "missing").sum())))
        lines.append(self._kv_line("Method-dataset aggregates", len(summary_df)))
        lines.append(self._kv_line("Successful aggregates", int((summary_df["status"] == "success").sum())))
        lines.append(self._kv_line("Methods represented", int(summary_df["method"].nunique())))
        lines.append(self._kv_line("Datasets represented", int(summary_df["dataset"].nunique())))
        lines.append(self._kv_line("Classification aggregates", int(len(clf_df))))
        lines.append(self._kv_line("Regression aggregates", int(len(reg_df))))
        if "feature_set_count" in success_df.columns and not success_df.empty:
            lines.append(
                self._kv_line(
                    "Mean saved runs per successful pair",
                    f"{success_df['feature_set_count'].mean():.2f}",
                )
            )
        lines.append("")
        lines.append("All report tables summarise saved feature-set means at the method-dataset level.")
        lines.append("")

        lines.extend(self._section_header("1. Overall Method Rankings"))
        lines.append("Classification is ranked by mean AUC (higher is better).")
        if not clf_df.empty:
            clf_rank = (
                clf_df.groupby("method")
                .agg(
                    mean_auc=("auc_mean", "mean"),
                    std_auc=("auc_mean", "std"),
                    n_datasets=("dataset", "count"),
                )
                .sort_values(["mean_auc", "method"], ascending=[False, True])
                .reset_index()
            )
            lines.extend(self._ranking_table(clf_rank, "mean_auc", "std_auc", higher_is_better=True))
        else:
            lines.append("  No classification results available.")
        lines.append("")
        lines.append("Regression is ranked by mean NRMSE (lower is better).")
        if not reg_df.empty:
            reg_rank = (
                reg_df.groupby("method")
                .agg(
                    mean_nrmse=("nrmse_mean", "mean"),
                    std_nrmse=("nrmse_mean", "std"),
                    n_datasets=("dataset", "count"),
                )
                .sort_values(["mean_nrmse", "method"], ascending=[True, True])
                .reset_index()
            )
            lines.extend(self._ranking_table(reg_rank, "mean_nrmse", "std_nrmse", higher_is_better=False))
        else:
            lines.append("  No regression results available.")
        lines.append("")

        lines.extend(self._section_header("2. Per-Dataset Accuracy"))
        lines.append("Classification accuracy by dataset and method. Cell values are mean ± std across saved runs.")
        lines.append("")
        lines.extend(
            self._pivot_table(
                clf_df,
                value_col="accuracy_mean",
                fmt=".4f",
                higher_is_better=True,
                std_col="accuracy_std",
            )
        )
        lines.append("")

        lines.extend(self._section_header("3. Per-Dataset AUC"))
        lines.append("Classification AUC by dataset and method. Cell values are mean ± std across saved runs.")
        lines.append("")
        lines.extend(
            self._pivot_table(
                clf_df,
                value_col="auc_mean",
                fmt=".4f",
                higher_is_better=True,
                std_col="auc_std",
            )
        )
        lines.append("")

        lines.extend(self._section_header("4. Baseline Comparison"))
        lines.append("Baseline scores are from XGBoost trained on the original dataset features.")
        lines.append("")
        lines.append("Classification (AUC)")
        lines.extend(
            self._baseline_table(
                clf_df,
                metric_col="auc_mean",
                baseline_col="baseline_auc_mean",
                higher_is_better=True,
            )
        )
        lines.append("")
        lines.append("Regression (NRMSE)")
        lines.extend(
            self._baseline_table(
                reg_df,
                metric_col="nrmse_mean",
                baseline_col="baseline_nrmse_mean",
                higher_is_better=False,
            )
        )
        lines.append("")

        lines.extend(self._section_header("5. Axis-Stratified Analysis"))
        axes_config = [
            ("Sample Size", "sample_size_axis"),
            ("Dimensionality", "dimensionality_axis"),
            ("Feature Type", "feature_type_axis"),
            ("LLM Familiarity", "llm_familiarity"),
        ]
        for axis_label, axis_col in axes_config:
            lines.append("")
            lines.append(axis_label)
            lines.extend(self._axis_table(clf_df, reg_df, axis_col, axis_label))
        lines.append("")

        lines.extend(self._section_header("Method Notes", char="-"))
        lines.append("- XGBoost configuration: n_estimators=100, max_depth=6, learning_rate=0.1")
        lines.append(f"- Evaluation uses {self.n_seeds} random seeds with test_size={self.test_size}")
        lines.append("- NRMSE is defined as RMSE / (y_max - y_min)")
        lines.append("- Multi-class ROC AUC uses one-vs-rest macro averaging")
        lines.append("- An asterisk (*) marks the best value in a dataset row")
        lines.append("- Baseline deltas are computed against XGBoost on the original feature set")
        lines.append("- Feature matrices must preserve full-dataset row count and row order")
        lines.append("- When multiple saved runs exist for a method-dataset pair, tables report their mean score")
        lines.append("- Sample size bands: Small < 500, Medium 500-5000, Large > 5000")
        lines.append("- Dimensionality bands: Low < 10, Medium 10-30, High > 30")
        lines.append("- Feature type bands: Numerical (<20% categorical), Mixed, Categorical (>80% categorical)")
        lines.append("=" * REPORT_WIDTH)
        return "\n".join(lines)

    def save_results(
        self,
        results_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        report_str: str,
    ) -> None:
        """
        Persist raw results, aggregated means, and report text to output_dir.

        Files written:
        - results.csv               : per-feature-set results table
        - results_mean.csv          : aggregated method/dataset means
        - benchmark_report_{ts}.txt : formatted report

        Parameters
        ----------
        results_df : pd.DataFrame
            Output of run_evaluation().
        summary_df : pd.DataFrame
            Output of aggregate_results().
        report_str : str
            Output of generate_report().
        """
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, "results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info("Results CSV saved to: %s", csv_path)

        mean_csv_path = os.path.join(self.output_dir, "results_mean.csv")
        summary_df.to_csv(mean_csv_path, index=False)
        logger.info("Mean results CSV saved to: %s", mean_csv_path)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"benchmark_report_{ts}.txt")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report_str)
        logger.info("Benchmark report saved to: %s", report_path)

    @staticmethod
    def _section_header(title: str, char: str = "=") -> List[str]:
        """Return a consistently styled section header."""
        return [char * REPORT_WIDTH, title, char * REPORT_WIDTH]

    @staticmethod
    def _kv_line(label: str, value) -> str:
        """Format a summary key-value line with stable alignment."""
        return f"{label:<42}: {value}"

    @staticmethod
    def _ranking_table(
        rank_df: pd.DataFrame,
        mean_col: str,
        std_col: str,
        *,
        higher_is_better: bool,
    ) -> List[str]:
        """Build a method ranking table with display-friendly labels."""
        lines: List[str] = []
        if rank_df.empty:
            lines.append("  (no data)")
            return lines

        display_names = [_display_method_name(method) for method in rank_df["method"]]
        method_w = max(32, max(len(name) for name in display_names) + 2)
        score_label = "Mean AUC" if higher_is_better else "Mean NRMSE"

        lines.append(f"  {'Rank':<6}{'Method':<{method_w}}{score_label:<18}{'Datasets'}")
        lines.append("  " + "-" * (method_w + 30))

        for rank, (_, row) in enumerate(rank_df.iterrows(), start=1):
            std_value = row[std_col] if not pd.isna(row[std_col]) else 0.0
            method_name = _display_method_name(row["method"])
            note = "  top" if rank == 1 else ""
            lines.append(
                f"  {rank:<6}{method_name:<{method_w}}"
                f"{row[mean_col]:.4f} +/- {std_value:.4f}   "
                f"{int(row['n_datasets'])}{note}"
            )
        return lines

    @staticmethod
    def _baseline_table(
        df: pd.DataFrame,
        *,
        metric_col: str,
        baseline_col: str,
        higher_is_better: bool,
    ) -> List[str]:
        """Build a baseline comparison table for one task family."""
        lines: List[str] = []
        if df.empty:
            lines.append("  No results available.")
            return lines

        datasets = sorted(df["dataset"].unique())
        methods = _sort_methods(list(df["method"].unique()))
        method_labels = {_display_method_name(method): method for method in methods}
        col_labels = [_display_method_name(method) for method in methods]

        ds_w = max(24, max(len(str(dataset)) for dataset in datasets) + 2)
        col_w = max(16, max(len(label) for label in col_labels) + 2)

        header = f"  {'Dataset':<{ds_w}}{'Baseline':>{col_w}}"
        header += "".join(f"{label:>{col_w}}" for label in col_labels)
        header += f"{'Best Delta':>{col_w}}"
        lines.append(header)
        lines.append("  " + "-" * (ds_w + col_w * (len(col_labels) + 2)))

        for dataset_name in datasets:
            ds_rows = df[df["dataset"] == dataset_name]
            baseline_values = ds_rows[baseline_col].dropna()
            baseline_value = baseline_values.iloc[0] if not baseline_values.empty else None
            baseline_text = f"{baseline_value:.4f}" if baseline_value is not None else "N/A"

            method_values = {}
            for label, method in method_labels.items():
                method_row = ds_rows[ds_rows["method"] == method]
                if not method_row.empty and not pd.isna(method_row[metric_col].iloc[0]):
                    method_values[label] = method_row[metric_col].iloc[0]

            best_method = None
            if method_values:
                comparator = max if higher_is_better else min
                best_method = comparator(method_values, key=method_values.__getitem__)

            best_delta = None
            cells = [baseline_text]
            for label in col_labels:
                if label not in method_values:
                    cells.append("N/A")
                    continue

                value = method_values[label]
                marker = "*" if label == best_method else " "
                cells.append(f"{value:.4f}{marker}")
                if baseline_value is not None:
                    delta = value - baseline_value
                    if best_delta is None:
                        best_delta = delta
                    elif higher_is_better and delta > best_delta:
                        best_delta = delta
                    elif not higher_is_better and delta < best_delta:
                        best_delta = delta

            cells.append(f"{best_delta:+.4f}" if best_delta is not None else "N/A")
            lines.append(f"  {dataset_name:<{ds_w}}" + "".join(f"{cell:>{col_w}}" for cell in cells))

        return lines

    @staticmethod
    def _find_target_column(df: pd.DataFrame) -> str:
        """
        Locate the target column in a DataFrame.

        Checks for columns named "target", "class", or "label"
        (case-insensitive) before falling back to the last column.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset including the target column.

        Returns
        -------
        str
            Name of the target column.
        """
        priority_names = {"target", "class", "label"}
        for col in df.columns:
            if col.lower() in priority_names:
                return col
        return df.columns[-1]

    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """
        In-place imputation: median for numeric columns, mode for object/category.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (no target column).

        Returns
        -------
        pd.DataFrame
            Imputed DataFrame.
        """
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_vals = df[col].mode()
                    fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "missing"
                    df[col].fillna(fill_val, inplace=True)
        return df

    @staticmethod
    def _pivot_table(
        df: pd.DataFrame,
        value_col: str,
        fmt: str = ".4f",
        higher_is_better: bool = True,
        std_col: Optional[str] = None,
    ) -> List[str]:
        """
        Build a pivot table (datasets as rows, methods as columns).
        Best value per row is marked with *.
        If std_col is provided, cells show value+/-std.
        """
        lines: List[str] = []
        if df.empty or value_col not in df.columns:
            lines.append("  (no data)")
            return lines

        sub = df[["dataset", "method", value_col]].dropna(subset=[value_col])
        if sub.empty:
            lines.append("  (no data)")
            return lines

        pivot = sub.pivot_table(
            index="dataset", columns="method", values=value_col, aggfunc="mean"
        )
        pivot = pivot.reindex(columns=_sort_methods(list(pivot.columns)))

        # Build std pivot if requested
        std_pivot = None
        if std_col and std_col in df.columns:
            std_sub = df[["dataset", "method", std_col]].dropna(subset=[std_col])
            if not std_sub.empty:
                std_pivot = std_sub.pivot_table(
                    index="dataset", columns="method", values=std_col, aggfunc="mean"
                )
                std_pivot = std_pivot.reindex(columns=pivot.columns)

        methods = list(pivot.columns)
        method_labels = [_display_method_name(method) for method in methods]
        # Width: value(6) + +/-std(7) + marker(1) + padding(2) = 16 min
        cell_w = 16 if std_pivot is not None else 10
        col_w = max(cell_w, max(len(label) for label in method_labels) + 2)
        ds_w  = max(22, max(len(str(d)) for d in pivot.index) + 2)

        header = f"  {'Dataset':<{ds_w}}" + "".join(f"{label:>{col_w}}" for label in method_labels)
        lines.append(header)
        lines.append("  " + "-" * (ds_w + col_w * len(methods)))

        for dataset, row_data in pivot.iterrows():
            valid_vals = {m: row_data[m] for m in methods if not pd.isna(row_data[m])}
            if valid_vals:
                best_m = max(valid_vals, key=valid_vals.__getitem__) if higher_is_better                          else min(valid_vals, key=valid_vals.__getitem__)
            else:
                best_m = None

            cells = []
            for m in methods:
                val = row_data[m]
                if pd.isna(val):
                    cells.append("N/A")
                else:
                    marker = "*" if m == best_m else " "
                    if std_pivot is not None and m in std_pivot.columns:
                        std_val = std_pivot.loc[dataset, m] if dataset in std_pivot.index else float("nan")
                        std_str = f"+/-{std_val:.4f}" if not pd.isna(std_val) else ""
                        cell = f"{val:{fmt}}{std_str}{marker}"
                    else:
                        cell = f"{val:{fmt}}{marker}"
                    cells.append(cell)

            lines.append(
                f"  {str(dataset):<{ds_w}}" + "".join(f"{c:>{col_w}}" for c in cells)
            )

        lines.append("  (* = best value in row)")
        return lines

    @staticmethod
    def _axis_table(
        clf_df: pd.DataFrame,
        reg_df: pd.DataFrame,
        axis_col: str,
        axis_label: str,
    ) -> List[str]:
        """
        Build axis-stratified summary tables for classification (AUC) and
        regression (NRMSE).

        Parameters
        ----------
        clf_df : pd.DataFrame
            Successful classification results.
        reg_df : pd.DataFrame
            Successful regression results.
        axis_col : str
            Column name in results_df to stratify by.
        axis_label : str
            Human-readable label for the axis.

        Returns
        -------
        list of str
            Formatted table lines.
        """
        lines: List[str] = []

        def _sub_table(sub_df: pd.DataFrame, metric_col: str, task_label: str) -> List[str]:
            t_lines: List[str] = []
            if sub_df.empty or axis_col not in sub_df.columns:
                return t_lines
            grp = (
                sub_df.dropna(subset=[metric_col])
                .groupby(["method", axis_col])[metric_col]
                .mean()
                .unstack(axis_col)
            )
            if grp.empty:
                return t_lines

            grp = grp.reindex(index=_sort_methods(list(grp.index)))
            axis_vals = list(grp.columns)
            methods = list(grp.index)
            col_w = 14
            ds_w = max(30, max(len(_display_method_name(method)) for method in methods) + 2)

            t_lines.append(f"  [{task_label} - mean {metric_col}]")
            header = f"  {'Method':<{ds_w}}" + "".join(f"{av:>{col_w}}" for av in axis_vals)
            t_lines.append(header)
            t_lines.append("  " + "-" * (ds_w + col_w * len(axis_vals)))

            for method in methods:
                cells = []
                for av in axis_vals:
                    val = grp.loc[method, av] if av in grp.columns else np.nan
                    cells.append(f"{val:.4f}" if not pd.isna(val) else "N/A")
                t_lines.append(
                    f"  {_display_method_name(method):<{ds_w}}" + "".join(f"{c:>{col_w}}" for c in cells)
                )
            return t_lines

        lines.extend(_sub_table(clf_df, "auc_mean", "Classification"))
        lines.extend(_sub_table(reg_df, "nrmse_mean", "Regression"))

        if not lines:
            lines.append("  (no data for this axis)")

        return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM feature engineering methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        default="./features",
        help="Root directory containing method sub-folders with feature CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory where results.csv and the benchmark report are written.",
    )
    parser.add_argument(
        "--datasets_dir",
        default="./datasets",
        help="Directory containing original dataset CSVs.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds to average metrics over.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = BenchmarkEvaluator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        datasets_dir=args.datasets_dir,
        test_size=args.test_size,
        n_seeds=args.n_seeds,
    )

    results_df = evaluator.run_evaluation()
    summary_df = evaluator.aggregate_results(results_df)
    report = evaluator.generate_report(results_df, summary_df)
    evaluator.save_results(results_df, summary_df, report)
    print(report)
