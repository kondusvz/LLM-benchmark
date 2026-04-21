"""Shared dataset loading, validation, and feature-output utilities.

This module is the single source of truth for benchmark dataset metadata,
cleaning, validation, and feature CSV output paths.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads named benchmark datasets from CSV files.

    A metadata registry maps datasets to their task type, known target
    column name, approximate feature count, and optional cleaning config.

    Preprocessing applied on load:
    - Column dropping: removes columns listed in the ``drop_cols`` registry.
    - NaN handling: drops rows where the target is NaN, then imputes
      features (median for numeric, mode for categorical/object).
    - Target reordering: ensures the target column is last (required by
      some downstream tools, e.g. FeatLLM's utils.get_dataset).
    - Label encoding (in ``load`` / ``load_dataframe`` only): all
      non-numeric X columns are encoded; target is encoded when
      non-numeric.

    The raw CSV on disk is **never modified**.  All cleaning happens
    in-memory, or callers can request a cleaned copy via
    ``export_clean_csv()``.

    Usage
    -----
    loader = DatasetLoader()

    # For methods that need DataFrames (e.g. LLM-FE, CAAFE):
    X_df, y, task_type, feature_names = loader.load_dataframe("diabetes")

    # For methods that need numpy arrays (e.g. OCTree):
    X, y, task_type = loader.load("diabetes")

    # For methods that need a cleaned CSV on disk without label-encoding
    # (e.g. FeatLLM repo utils that read CSV directly):
    clean_path = loader.export_clean_csv("Titanic-Dataset", dest_dir="./data")
    """

    DATASET_METADATA: Dict[str, Tuple[str, str, int]] = {
        # Classification - High familiarity
        "breast-w":              ("classification", "Class",          9),
        "diabetes":              ("classification", "Outcome",        8),
        "adult":                 ("classification", "income",         14),
        "Titanic-Dataset":       ("classification", "Survived",       7),
        "mushroom":              ("classification", "poisonous",     22),
        # Classification - Medium familiarity
        "bank":                  ("classification", "Class",         16),
        "blood":                 ("classification", "Class",          4),
        "credit-g":              ("classification", "class",         20),
        "car":                   ("classification", "class",          6),
        "spambase":              ("classification", "spam",         57),
        "nursery":               ("classification", "class",          8),
        # Classification - Low familiarity
        "Iris":                  ("classification", "Species",        4),
        "wine":                  ("classification", "quality",       12),
        "ionosphere":            ("classification", "Class",         34),
        "MAGIC-gt":              ("classification", "class",         10),
        # Regression
        "housing":               ("regression",     "median_house_value", 9),  # California
        "HousingData":           ("regression",     "MEDV",              13),  # Boston
        "abalone":               ("regression",     "Rings",              8),
        "energy-efficiency":     ("regression",     "Y1",                 8),
        "bike-sharing":          ("regression",     "cnt",               12),
        "yacht-hydro":           ("regression",     "ResiduaryResistance", 6),
    }

    DATASET_FAMILIARITY: Dict[str, str] = {
        # Classification - High familiarity
        "breast-w":        "High",
        "diabetes":        "High",
        "adult":           "High",
        "Titanic-Dataset": "High",
        "mushroom":        "High",
        # Classification - Medium familiarity
        "bank":            "Medium",
        "blood":           "Medium",
        "credit-g":        "Medium",
        "car":             "Medium",
        "spambase":        "Medium",
        "nursery":         "Medium",
        # Classification - Low familiarity
        "Iris":            "Low",
        "wine":            "Low",
        "ionosphere":      "Low",
        "MAGIC-gt":        "Low",
        # Regression
        "housing":            "High",
        "HousingData":        "High",
        "abalone":            "Medium",
        "energy-efficiency":  "High",
        "bike-sharing":       "High",
        "yacht-hydro":        "Low",
    }

    DROP_COLS: Dict[str, List[str]] = {
        "Titanic-Dataset": ["PassengerId", "Name", "Ticket", "Cabin"],
        "Iris":            ["Id"],   # row identifier - no predictive value
        "energy-efficiency": ["Y2"],     # second regression target not used here
        "bike-sharing":      ["dteday"], # date stamp duplicates richer calendar fields
    }

    DATASET_DESCRIPTIONS: Dict[str, str] = {
        # ------------------------------------------------------------------
        # Classification - High familiarity
        # ------------------------------------------------------------------
        "breast-w": (
            "Wisconsin Breast Cancer dataset. Each row represents a digitised "
            "fine-needle aspirate (FNA) of a breast mass from a single patient. "
            "Features describe characteristics of the cell nuclei present in the "
            "image, including clump thickness, uniformity of cell size and shape, "
            "marginal adhesion, single epithelial cell size, bare nuclei, bland "
            "chromatin, normal nucleoli, and mitoses - all rated on a scale of 1 "
            "to 10. The target is whether the tumour is benign or malignant."
        ),
        "diabetes": (
            "Pima Indians Diabetes dataset. Each row represents a female patient "
            "of Pima Indian heritage, aged at least 21. Features include the number "
            "of pregnancies, plasma glucose concentration, diastolic blood pressure, "
            "triceps skin fold thickness, 2-hour serum insulin, body mass index "
            "(BMI), diabetes pedigree function (a measure of genetic risk), and age. "
            "The target is whether the patient was diagnosed with diabetes within "
            "five years of the examination."
        ),
        "adult": (
            "Adult Census Income dataset, also known as the Census Income dataset. "
            "Each row represents an individual drawn from the 1994 US Census. "
            "Features include age, workclass, education level and years of education, "
            "marital status, occupation, relationship, race, sex, capital gain, "
            "capital loss, hours worked per week, and native country. The target is "
            "whether the individual's annual income exceeds $50,000."
        ),
        "Titanic-Dataset": (
            "Titanic passenger survival dataset. Each row represents a passenger "
            "aboard the RMS Titanic. Features include passenger class (1st, 2nd, "
            "or 3rd), sex, age, number of siblings or spouses aboard (SibSp), "
            "number of parents or children aboard (Parch), and the fare paid. "
            "Identifier columns (name, ticket, cabin, passenger ID) have been "
            "removed. The target is whether the passenger survived the sinking."
        ),
        # ------------------------------------------------------------------
        # Classification - Medium familiarity
        # ------------------------------------------------------------------
        "bank": (
            "Bank Marketing dataset from a Portuguese banking institution. Each row "
            "represents a phone contact with a client during a direct marketing "
            "campaign. Features include client demographics (age, job, marital "
            "status, education, credit default status, housing loan, personal loan), "
            "contact details (contact type, month, day of week, call duration), and "
            "campaign information (number of contacts in this campaign, days since "
            "last contact, previous campaign contacts and outcome). The target is "
            "whether the client subscribed to a term deposit."
        ),
        "blood": (
            "Blood Transfusion Service Center dataset. Each row represents a blood "
            "donor. Features are: Recency (months since last donation), Frequency "
            "(total number of donations), Monetary (total blood donated in cubic "
            "centimetres), and Time (months since first donation). The target is "
            "whether the donor donated blood in March 2007."
        ),
        "credit-g": (
            "German Credit dataset. Each row represents a loan applicant at a "
            "German bank. Features describe the applicant's financial situation and "
            "personal background, including checking account status, loan duration, "
            "credit history, loan purpose, credit amount, savings account, "
            "employment duration, instalment rate, personal status and sex, other "
            "debtors, residence duration, property owned, age, other instalment "
            "plans, housing, number of existing credits, job type, number of "
            "dependants, telephone, and foreign worker status. The target is whether "
            "the applicant is a good or bad credit risk."
        ),
        "german_credit_data": (
            "German Credit Data dataset. Each row represents a bank loan applicant. "
            "Features include age, sex, job type, housing status (own, free, or "
            "rent), savings account level, checking account level, credit amount, "
            "loan duration in months, and loan purpose. The target is the purpose "
            "of the credit application (e.g. car, furniture, education, business)."
        ),
        "car": (
            "Car Evaluation dataset. Each row represents a car evaluated along "
            "several dimensions. Features are buying price, maintenance price, "
            "number of doors, passenger capacity, luggage boot size, and estimated "
            "safety rating. The target is the overall acceptability of the car "
            "(unacceptable, acceptable, good, or very good)."
        ),
        "spambase": (
            "Spambase dataset from Hewlett-Packard Labs. Each row represents an "
            "email message. Features are the relative frequencies of 48 specific "
            "words and 6 specific characters in the email, the average, longest, "
            "and total length of uninterrupted sequences of capital letters, giving "
            "57 features in total. The target is whether the email is spam."
        ),
        # ------------------------------------------------------------------
        # Classification - Low familiarity
        # ------------------------------------------------------------------
        "Iris": (
            "Iris flower dataset. Each row represents a single iris flower specimen. "
            "Features are sepal length, sepal width, petal length, and petal width, "
            "all measured in centimetres. The target is the species of iris: Setosa, "
            "Versicolor, or Virginica."
        ),
        "wine": (
            "Wine Quality dataset for red Vinho Verde wine from northern Portugal. "
            "Each row represents a wine sample. Features are physicochemical "
            "measurements: fixed acidity, volatile acidity, citric acid, residual "
            "sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, "
            "pH, sulphates, alcohol content, and wine color. The target is a wine "
            "quality score assigned by experts, here treated as a 7-class "
            "classification target with labels 3 through 9."
        ),
        "mushroom": (
            "Mushroom dataset. Each row represents one hypothetical mushroom "
            "described by categorical attributes such as cap shape, cap surface, "
            "cap color, bruising, odor, gill attachment, gill spacing, gill size, "
            "gill color, stalk shape, stalk root, stalk surfaces and colors, veil "
            "type and color, ring number and type, spore print color, population, "
            "and habitat. The target is whether the mushroom is edible or poisonous."
        ),
        "nursery": (
            "Nursery dataset derived from a hierarchical decision model for ranking "
            "applications to nursery schools. Each row represents one application. "
            "Features describe the parents, nursery status, family form, number of "
            "children, housing situation, financial standing, social conditions, "
            "and health recommendation. The target is the recommended priority "
            "class for the application."
        ),
        "ionosphere": (
            "Ionosphere radar dataset. Each row represents a radar return from the "
            "ionosphere measured by a phased array system. Features are 34 "
            "continuous attributes derived from the radar signals. The target is "
            "whether the return shows a good structured signal or a bad return."
        ),
        "MAGIC-gt": (
            "MAGIC Gamma Telescope dataset. Each row represents an event recorded "
            "by an imaging atmospheric Cherenkov telescope. Features describe the "
            "shape, orientation, and intensity distribution of the detected shower "
            "image. The target is whether the event was caused by a gamma ray or "
            "by hadronic background."
        ),
        # ------------------------------------------------------------------
        # Regression
        # ------------------------------------------------------------------
        "housing": (
            "California Housing dataset derived from the 1990 US Census. Each row "
            "represents a census block group in California. Features include median "
            "income, housing median age, average number of rooms per household, "
            "average number of bedrooms per household, population, average household "
            "occupancy, ocean proximity, and geographical coordinates (latitude and "
            "longitude). The target is the median house value for households within "
            "the block group."
        ),
        "HousingData": (
            "Boston Housing dataset collected by the US Census Service for the "
            "Boston, Massachusetts area. Each row represents a town or suburb. "
            "Features include per capita crime rate, proportion of residential land "
            "zoned for large lots, proportion of non-retail business acres, Charles "
            "River adjacency, nitric oxides concentration, average number of rooms "
            "per dwelling, proportion of owner-occupied units built before 1940, "
            "weighted distance to employment centres, accessibility to radial "
            "highways, property tax rate, pupil-teacher ratio, proportion of Black "
            "residents, and percentage of lower-status population. The target is "
            "the median value of owner-occupied homes in thousands of dollars."
        ),
        "abalone": (
            "Abalone dataset from marine biology. Each row represents an individual "
            "abalone (a type of sea snail). Features include sex (male, female, or "
            "infant), length, diameter, height, whole weight, shucked weight "
            "(meat only), viscera weight (gut weight after bleeding), and shell "
            "weight after drying. The target is the number of rings in the shell, "
            "which is used to determine age (rings + 1.5 gives age in years)."
        ),
        "energy-efficiency": (
            "Energy Efficiency dataset for building simulation. Each row represents "
            "a building design variant. Features include relative compactness, "
            "surface area, wall area, roof area, overall height, orientation, "
            "glazing area, and glazing area distribution. The original dataset has "
            "two targets: heating load (Y1) and cooling load (Y2). This benchmark "
            "uses heating load (Y1) as the regression target and drops Y2."
        ),
        "bike-sharing": (
            "Bike Sharing dataset. Each row represents one hour of bike rental "
            "activity. Features include season, year, month, hour, holiday flag, "
            "day of week, working day flag, weather situation, temperature, "
            "feels-like temperature, humidity, and windspeed. The raw date stamp "
            "is dropped because the calendar components are already present. The "
            "target is the total number of rented bikes in that hour."
        ),
        "yacht-hydro": (
            "Yacht Hydrodynamics dataset. Each row represents one hull-design and "
            "operating-point configuration for a sailing yacht. Features are the "
            "longitudinal position of the center of buoyancy, prismatic "
            "coefficient, length-displacement ratio, beam-draught ratio, "
            "length-beam ratio, and Froude number. The target is residuary "
            "resistance per unit weight of displacement."
        ),
    }

    def read_and_clean(
        self,
        dataset_name: str,
        datasets_dir: str = "./datasets",
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        Read a raw CSV and return a cleaned DataFrame (no label-encoding).

        Cleaning steps
        --------------
        1. Drop columns listed in ``DROP_COLS`` (if present in the CSV).
        2. Resolve the target column.
        3. Drop rows where the target is NaN.
        4. Impute feature NaNs (median / mode).
        5. Move the target column to the last position.

        Returns
        -------
        df_clean : pd.DataFrame
            Cleaned DataFrame with target as the last column.
        target_col : str
            Name of the target column.
        task_type : str
            ``"classification"`` or ``"regression"``.
        """
        csv_path = os.path.join(datasets_dir, f"{dataset_name}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        cols_to_drop = [
            c for c in self.DROP_COLS.get(dataset_name, []) if c in df.columns
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info("Dropped columns %s from '%s'", cols_to_drop, dataset_name)

        target_col = self._resolve_target_column(df, dataset_name)
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in '{dataset_name}'. "
                f"Available columns: {list(df.columns)}"
            )

        n_before = len(df)
        df = df.dropna(subset=[target_col])
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info(
                "Dropped %d rows with NaN target in '%s'", n_dropped, dataset_name
            )

        feature_cols = [c for c in df.columns if c != target_col]
        df[feature_cols] = self._impute(df[feature_cols])

        other_cols = [c for c in df.columns if c != target_col]
        df = df[other_cols + [target_col]].reset_index(drop=True)

        task_type = self._resolve_task_type(dataset_name, df[target_col])

        logger.info(
            "Cleaned '%s': %d rows, %d features, target='%s', task='%s'",
            dataset_name, len(df), len(other_cols), target_col, task_type,
        )
        return df, target_col, task_type

    def export_clean_csv(
        self,
        dataset_name: str,
        datasets_dir: str = "./datasets",
        dest_dir: str = "./data",
        force: bool = False,
    ) -> str:
        """
        Write a cleaned (but not label-encoded) CSV to *dest_dir*.

        Useful for tools like FeatLLM's ``utils.get_dataset`` that read
        CSVs directly.  The raw CSV in *datasets_dir* is never modified.

        Parameters
        ----------
        dataset_name : str
        datasets_dir : str
            Where the original Kaggle CSV lives.
        dest_dir : str
            Where the cleaned copy should be written.
        force : bool
            If True, overwrite even if the destination already exists.

        Returns
        -------
        str
            Absolute path of the written CSV.
        """
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{dataset_name}.csv")

        if os.path.isfile(dest_path) and not force:
            logger.info("Clean CSV already exists: %s", dest_path)
            return os.path.abspath(dest_path)

        df, target_col, _ = self.read_and_clean(dataset_name, datasets_dir)
        df.to_csv(dest_path, index=False)
        logger.info("Clean CSV written -> %s", dest_path)
        return os.path.abspath(dest_path)

    # Per-dataset overrides for detect_categorical().
    #
    # The generic heuristic flags integer columns with <=10 unique values as
    # categorical.  This is wrong for datasets where those integers are
    # continuous measurements or counts rather than nominal codes.
    #
    # For each dataset, None means "force all integer columns to numeric".
    # An explicit dict means "apply these specific column overrides".
    #
    # breast-w:  all 9 features are 1-10 ordinal ratings - numeric
    # diabetes:  pregnancies (0-17), glucose, BP etc. - all continuous counts
    # blood:     Recency/Frequency/Monetary/Time - all continuous counts
    # spambase:  57 float/int frequency features - all continuous
    # Titanic:   Pclass (1/2/3) is genuinely ordinal categorical; Survived
    #            is the target so not in X; SibSp/Parch are counts -> numeric
    _IS_CAT_OVERRIDES: Dict[str, Optional[Dict[str, bool]]] = {
        "breast-w":        None,   # all features -> numeric
        "diabetes":        None,   # all features -> numeric
        "blood":           None,   # all features -> numeric
        "spambase":        None,   # all features -> numeric
        "Titanic-Dataset": {"Pclass": True, "SibSp": False, "Parch": False},
        "bike-sharing": {
            "season": True,
            "yr": True,
            "mnth": True,
            "hr": True,
            "holiday": True,
            "weekday": True,
            "workingday": True,
            "weathersit": True,
        },
        "energy-efficiency": {"X6": True, "X8": True},
    }

    @staticmethod
    def detect_categorical(
        df: pd.DataFrame,
        target_col: str,
        int_nunique_threshold: int = 10,
        dataset_name: str = "",
    ) -> Dict[str, bool]:
        """
        Determine which feature columns are categorical.

        A column is categorical if it has object/string/bool dtype, or if
        it is integer-typed with <= *int_nunique_threshold* unique values.

        Per-dataset overrides in ``_IS_CAT_OVERRIDES`` take precedence:
        - ``None``  -> force every feature column to numeric (False)
        - ``dict``  -> apply the specified per-column overrides on top of
                      the heuristic result; unmentioned columns keep their
                      heuristic value

        Parameters
        ----------
        df : pd.DataFrame
            Full DataFrame (features + target).
        target_col : str
            Name of the target column (excluded from output).
        int_nunique_threshold : int
            Max unique values for an integer column to be considered
            categorical (default: 10).
        dataset_name : str
            Dataset name used to look up per-dataset overrides.

        Returns
        -------
        dict mapping column name -> bool
        """
        is_cat: Dict[str, bool] = {}
        for col in df.columns:
            if col == target_col:
                continue
            dt = df[col].dtype
            if dt == object or pd.api.types.is_string_dtype(dt) or str(dt) == "bool":
                is_cat[col] = True
            elif pd.api.types.is_integer_dtype(dt) and df[col].nunique() <= int_nunique_threshold:
                is_cat[col] = True
            else:
                is_cat[col] = False

        if dataset_name in DatasetLoader._IS_CAT_OVERRIDES:
            override = DatasetLoader._IS_CAT_OVERRIDES[dataset_name]
            if override is None:
                is_cat = {col: False for col in is_cat}
            else:
                is_cat.update(override)

        return is_cat

    def load(
        self,
        dataset_name: str,
        datasets_dir: str = "./datasets",
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load and preprocess a dataset, returning numpy arrays.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (without .csv extension).
        datasets_dir : str, optional
            Directory containing the CSV files (default: "./datasets").

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features), dtype float32
        y : np.ndarray, shape (n_samples,)
        task_type : str
            "classification" or "regression".
        """
        X_df, y, task_type, _ = self.load_dataframe(dataset_name, datasets_dir)
        return X_df.to_numpy(dtype=np.float32), y.to_numpy(), task_type

    def load_dataframe(
        self,
        dataset_name: str,
        datasets_dir: str = "./datasets",
    ) -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
        """
        Load and preprocess a dataset, returning a DataFrame with column names.

        Suitable for methods that need feature names in prompts (LLM-FE, CAAFE).

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (without .csv extension).
        datasets_dir : str, optional
            Directory containing the CSV files (default: "./datasets").

        Returns
        -------
        X : pd.DataFrame
            Preprocessed feature matrix (all numeric after encoding).
        y : pd.Series
            Target variable.
        task_type : str
            "classification" or "regression".
        feature_names : list of str
            Column names of X.
        """
        df, target_col, task_type = self.read_and_clean(dataset_name, datasets_dir)

        y_raw = df[target_col].copy()
        X_df = df.drop(columns=[target_col]).copy()

        logger.info(
            "Task type: %s, Target: %s, Classes/Range: %s",
            task_type, target_col,
            str(y_raw.nunique()) if task_type == "classification"
            else f"{y_raw.min():.2f}-{y_raw.max():.2f}",
        )

        # Label-encode non-numeric X columns
        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

        # Classification targets should always be remapped to 0..K-1 even if
        # the raw labels are numeric (e.g. wine quality 3..9 or Cover_Type 1..7).
        if task_type == "classification":
            y = pd.Series(
                LabelEncoder().fit_transform(y_raw.astype(str)),
                name=target_col,
            )
        else:
            y = y_raw

        feature_names = list(X_df.columns)
        logger.info(
            "Dataset '%s' loaded: %d rows, %d features, task='%s'",
            dataset_name, len(X_df), len(feature_names), task_type,
        )
        return X_df, y, task_type, feature_names

    def get_task_type(self, dataset_name: str) -> Optional[str]:
        """Return registered task type, or None if unknown."""
        meta = self.DATASET_METADATA.get(dataset_name)
        return meta[0] if meta is not None else None

    def get_target_column(self, dataset_name: str) -> Optional[str]:
        """Return registered target column name, or None if unknown."""
        meta = self.DATASET_METADATA.get(dataset_name)
        return meta[1] if meta is not None else None

    def get_description(self, dataset_name: str) -> Optional[str]:
        """
        Return a natural-language description of the dataset, or None if
        no description has been registered.

        These descriptions are used by LLM-based feature engineering methods
        (e.g. CAAFE) to ground feature generation in domain knowledge.
        """
        return self.DATASET_DESCRIPTIONS.get(dataset_name)

    def list_datasets(self, task_type: Optional[str] = None) -> List[str]:
        """
        Return registered dataset names, optionally filtered by task type.

        Parameters
        ----------
        task_type : str, optional
            If provided, only datasets with this task type are returned.

        Returns
        -------
        list of str
            Sorted dataset names from the shared registry.
        """
        datasets = sorted(self.DATASET_METADATA)
        if task_type is None:
            return datasets
        return [
            dataset_name
            for dataset_name in datasets
            if self.get_task_type(dataset_name) == task_type
        ]

    def get_llm_familiarity(self, dataset_name: str) -> str:
        """Return the benchmark's LLM-familiarity label for a dataset."""
        return self.DATASET_FAMILIARITY.get(dataset_name, "Unknown")

    def _resolve_target_column(self, df: pd.DataFrame, dataset_name: str) -> str:
        """
        Determine the target column for a loaded DataFrame.

        Resolution order:
        1. Registered target column from DATASET_METADATA (if present in df).
        2. A column named "target", "class", "Class", "label", "Label",
           "y", "output" (exact match).
        3. The last column.
        """
        registered = self.get_target_column(dataset_name)
        if registered and registered in df.columns:
            return registered

        # Heuristic fallback
        priority_names = ["target", "class", "Class", "label", "Label", "y", "output"]
        for candidate in priority_names:
            if candidate in df.columns:
                logger.debug("Target column resolved by heuristic: '%s'", candidate)
                return candidate

        # Last-column fallback
        logger.warning(
            "Target column for '%s' not in metadata or heuristic names; "
            "using last column '%s'.",
            dataset_name, df.columns[-1],
        )
        return df.columns[-1]

    def _resolve_task_type(self, dataset_name: str, y: pd.Series) -> str:
        """
        Determine task type from metadata or by inspecting the target.

        If the dataset is in the metadata registry, use the registered type.
        Otherwise, infer: if y is non-numeric or has < 20 unique values,
        treat as classification; else regression.
        """
        registered = self.get_task_type(dataset_name)
        if registered is not None:
            return registered

        # Infer from data
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 20:
            inferred = "classification"
        else:
            inferred = "regression"

        logger.warning(
            "Dataset '%s' not in metadata; inferred task_type='%s'.",
            dataset_name, inferred,
        )
        return inferred

    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values: median for numeric, mode for categorical.
        """
        df = df.copy()
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_vals = df[col].mode()
                    fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "missing"
                    df[col] = df[col].fillna(fill_val)
        return df


class DatasetValidator:
    """
    Validates a preprocessed dataset and returns a structured report.

    Works with both numpy arrays and DataFrames.

    Usage
    -----
    validator = DatasetValidator()
    report = validator.validate(X, y, "classification", "iris")
    if not report["is_valid"]:
        print(report["warnings"])
    """

    def validate(
        self,
        X,
        y,
        task_type: str,
        dataset_name: str,
    ) -> dict:
        """
        Validate a dataset and return a structured report dictionary.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix.
        y : np.ndarray or pd.Series
            Target vector.
        task_type : str
            "classification" or "regression".
        dataset_name : str
            Dataset name for logging purposes.

        Returns
        -------
        dict
            Keys: n_samples, n_features, has_nan_X, has_nan_y,
            n_classes, class_balance, variance_y, is_valid, warnings.
        """
        # Normalise to array/Series-like structures for checks
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        warn_list: List[str] = []
        is_valid = True

        n_samples = X_arr.shape[0]
        n_features = X_arr.shape[1] if X_arr.ndim == 2 else 1

        if isinstance(X, pd.DataFrame):
            has_nan_X = bool(X.isna().any().any())
        else:
            try:
                has_nan_X = bool(np.isnan(X_arr.astype(float)).any())
            except (ValueError, TypeError):
                has_nan_X = bool(pd.isna(pd.DataFrame(X_arr)).any().any())

        if isinstance(y, pd.Series):
            has_nan_y = bool(y.isna().any())
        else:
            try:
                has_nan_y = bool(np.isnan(y_arr.astype(float)).any())
            except (ValueError, TypeError):
                has_nan_y = bool(pd.isna(pd.Series(y_arr)).any())

        if has_nan_X:
            warn_list.append(f"[{dataset_name}] X contains NaN values after preprocessing.")
        if has_nan_y:
            warn_list.append(f"[{dataset_name}] y contains NaN values.")

        if n_samples < 10:
            warn_list.append(f"[{dataset_name}] Too few samples: {n_samples}.")
            is_valid = False
        if n_features < 1:
            warn_list.append(f"[{dataset_name}] No features.")
            is_valid = False

        n_classes: Optional[int] = None
        class_balance: Optional[dict] = None
        variance_y: Optional[float] = None

        if task_type == "classification":
            classes, counts = np.unique(y_arr, return_counts=True)
            n_classes = int(len(classes))
            total = len(y_arr)
            class_balance = {
                str(cls): round(float(cnt) / total, 4)
                for cls, cnt in zip(classes, counts)
            }
            if n_classes < 2:
                warn_list.append(f"[{dataset_name}] Only {n_classes} class(es).")
                is_valid = False
            if int(counts.min()) < 2:
                warn_list.append(
                    f"[{dataset_name}] Smallest class has {int(counts.min())} sample(s)."
                )
        elif task_type == "regression":
            variance_y = float(np.var(y_arr.astype(float)))
            if variance_y == 0.0:
                warn_list.append(f"[{dataset_name}] Target variance is zero.")

        if warn_list:
            for msg in warn_list:
                logger.warning(msg)
        else:
            logger.info(
                "[%s] Validation passed: %d samples, %d features.",
                dataset_name, n_samples, n_features,
            )

        return {
            "n_samples":     n_samples,
            "n_features":    n_features,
            "has_nan_X":     has_nan_X,
            "has_nan_y":     has_nan_y,
            "n_classes":     n_classes,
            "class_balance": class_balance,
            "variance_y":    variance_y,
            "is_valid":      is_valid,
            "warnings":      warn_list,
        }


class FeatureCSVWriter:
    """
    Writes a feature matrix to a CSV file.

    Handles both numpy arrays (auto-generates column names) and pandas
    DataFrames. Parent directories are created automatically.

    Usage
    -----
    writer = FeatureCSVWriter()
    writer.write(X_array, "./features/method_a/iris_features.csv")
    """

    def write(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        output_path: str,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Serialize a feature matrix to a CSV file.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix (2-D).
        output_path : str
            Destination file path.
        feature_names : list of str, optional
            Column names to assign.

        Returns
        -------
        str
            The output path written to.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_cols = X.shape[1]
            if feature_names is not None:
                if len(feature_names) != n_cols:
                    raise ValueError(
                        f"feature_names has {len(feature_names)} entries "
                        f"but X has {n_cols} columns."
                    )
                cols = feature_names
            else:
                cols = [f"feat_{i}" for i in range(n_cols)]
            df_out = pd.DataFrame(X, columns=cols)

        elif isinstance(X, pd.DataFrame):
            df_out = X.reset_index(drop=True)
            if feature_names is not None:
                if len(feature_names) != df_out.shape[1]:
                    raise ValueError(
                        f"feature_names has {len(feature_names)} entries "
                        f"but DataFrame has {df_out.shape[1]} columns."
                    )
                df_out.columns = feature_names
        else:
            raise TypeError(
                f"X must be ndarray or DataFrame, got {type(X).__name__}."
            )

        parent_dir = os.path.dirname(os.path.abspath(output_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        df_out.to_csv(output_path, index=False)
        logger.info(
            "Feature CSV written: %s  [shape: %d x %d]",
            output_path, df_out.shape[0], df_out.shape[1],
        )
        return output_path


def list_feature_run_dirs(
    output_dir: str,
    method: str,
    dataset_name: str,
) -> List[Path]:
    """
    Return existing per-run directories for a method/dataset pair.

    Expected layout:
        {output_dir}/{method}/{dataset_name}/{dataset_name}-{run_index}
    """
    dataset_dir = Path(output_dir) / method / dataset_name
    if not dataset_dir.is_dir():
        return []

    run_pattern = re.compile(rf"^{re.escape(dataset_name)}-(\d+)$")
    run_dirs: List[Tuple[int, Path]] = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        match = run_pattern.match(child.name)
        if match is None:
            continue
        run_dirs.append((int(match.group(1)), child))

    return [path for _, path in sorted(run_dirs, key=lambda item: item[0])]


def get_next_feature_run_index(
    output_dir: str,
    method: str,
    dataset_name: str,
) -> int:
    """Return the next available 1-based run index for a method/dataset pair."""
    run_dirs = list_feature_run_dirs(output_dir, method, dataset_name)
    if not run_dirs:
        return 1

    last_index = int(run_dirs[-1].name.rsplit("-", 1)[-1])
    return last_index + 1


def build_feature_run_dir(
    output_dir: str,
    method: str,
    dataset_name: str,
    run_index: int,
) -> Path:
    """
    Build the directory path for a single feature-generation run.

    Example
    -------
    build_feature_run_dir("./features", "caafe", "bank", 1)
    -> Path("./features/caafe/bank/bank-1")
    """
    return Path(output_dir) / method / dataset_name / f"{dataset_name}-{run_index}"


def build_feature_output_path(
    output_dir: str,
    method: str,
    dataset_name: str,
    run_index: int,
) -> Path:
    """Build the feature CSV path for a single feature-generation run."""
    return build_feature_run_dir(output_dir, method, dataset_name, run_index) / f"{dataset_name}_features.csv"
