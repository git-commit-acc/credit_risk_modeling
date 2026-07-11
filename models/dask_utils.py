# models/dask_utils.py
"""
Shared Dask infrastructure for the modeling layer.

This module exists to fix two systemic problems found across the original
models/*.py files:

  1. EVERY model (logistic, random_forest, xgboost, lightgbm) created its own
     `dask.distributed.Client(n_workers=4, threads_per_worker=2)` inside
     `fit()`, wrapped in a bare `try/except: pass`. Calling `Client()` when a
     client/cluster is already running either raises, silently connects to
     the wrong scheduler, or (most commonly here, since exceptions were
     swallowed) spins up an additional LocalCluster on top of the existing
     one. Across a pipeline that trains 5 base models + a meta-learner, and
     especially inside hyperparameter tuning (n_trials x cv_folds fits), this
     leaked dozens of clusters and was a major, unbounded source of RAM/CPU
     consumption -- directly contradicting the "minimal RAM usage" and
     "disk-based computation" requirements.

     Fix: a single process-wide Client is created lazily on first use and
     reused by every model. Call `close_dask_client()` once at the end of
     the pipeline (main.py does this in its `finally` block).

  2. `logistic.py`, `random_forest.py`, and `catboost_model.py` each
     re-implemented (with small inconsistencies) a categorical-encoding
     routine that called `X.compute()` unconditionally -- i.e. every single
     model call pulled the ENTIRE Dask DataFrame into driver RAM as pandas
     just to label-encode a handful of categorical columns. For a 47M-row
     servicing panel this is exactly the full-dataset materialization the
     refactor is meant to eliminate.

     Fix: `LazyCategoricalEncoder` below wraps `dask_ml.preprocessing.
     Categorizer` + `OrdinalEncoder`, both of which operate on Dask
     DataFrames using partition-wise/blockwise operations and a single
     distributed `unique()` pass to learn categories -- never a full
     `.compute()` of the feature matrix. It also handles the "unseen
     category at transform time" case that the original hand-rolled
     `LabelEncoder` loops handled ad hoc and inconsistently.
"""

import logging
import threading
from typing import List, Optional, Union

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster
from dask_ml.preprocessing import Categorizer, OrdinalEncoder

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_client: Optional[Client] = None


def get_dask_client(
    n_workers: int = 4,
    threads_per_worker: int = 2,
    memory_limit: str = "4GB",
    dashboard_address: Optional[str] = None,
) -> Client:
    """
    Return the process-wide Dask distributed Client, creating it on first
    call. Every model module should call this instead of constructing its
    own Client -- this is what makes it safe for main.py to train five base
    models, a meta-learner, and run hyperparameter tuning without spawning a
    new local cluster on every `fit()` call.
    """
    global _client
    with _client_lock:
        if _client is not None:
            try:
                # Cheap liveness check; if the scheduler died, fall through
                # and recreate rather than handing back a dead client.
                if _client.status == "running":
                    return _client
            except Exception:
                pass

        logger.info(
            f"Starting shared Dask LocalCluster "
            f"(workers={n_workers}, threads_per_worker={threads_per_worker}, "
            f"memory_limit={memory_limit})..."
        )
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=dashboard_address,
            processes=True,
        )
        _client = Client(cluster)
        logger.info(f"Dask dashboard: {_client.dashboard_link}")
        return _client


def close_dask_client() -> None:
    """Shut down the shared client/cluster. Call once at pipeline teardown."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
                logger.info("Shared Dask client closed.")
            except Exception as e:
                logger.warning(f"Error closing Dask client: {e}")
            finally:
                _client = None


def ensure_dask_dataframe(
    data: Union[dd.DataFrame, dd.Series, pd.DataFrame, pd.Series],
    npartitions: int = 8,
) -> Union[dd.DataFrame, dd.Series]:
    """Wrap a pandas object as Dask if needed; pass Dask objects through untouched."""
    if isinstance(data, (dd.DataFrame, dd.Series)):
        return data
    return dd.from_pandas(data, npartitions=npartitions)


def identify_categorical_columns(
    ddf: dd.DataFrame,
    max_cardinality_for_numeric: int = 20,
    _cardinality_sample_partitions: int = 3,
) -> List[str]:
    """
    Identify categorical columns WITHOUT materializing the full DataFrame.

    - object/category dtype columns are categorical by definition (no data
      scan needed -- this is metadata Dask already tracks).
    - low-cardinality numeric columns (e.g. flags encoded as 0/1/2) are
      inferred from a small sample of partitions rather than a full
      `.nunique()` pass over 47M rows, which would require a full shuffle.
    """
    # NOTE: on modern pandas (2.x), string columns created via
    # `dd.from_pandas`/`read_parquet` often come back as pandas'
    # StringDtype (`str(dtype) == "string"`), not the legacy "object"
    # dtype. Checking only for "object"/"category" silently missed these
    # entirely, so any dataset with StringDtype columns would treat them as
    # numeric and try to encode/scale raw text. Detect categorical-ness
    # structurally instead: anything that isn't numeric and isn't a
    # datetime is treated as categorical.
    dtype_cat_cols = [
        c for c, dt in ddf.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_datetime64_any_dtype(dt)
    ]

    numeric_cols = [
        c for c, dt in ddf.dtypes.items()
        if c not in dtype_cat_cols and pd.api.types.is_numeric_dtype(dt)
    ]

    if not numeric_cols:
        return dtype_cat_cols

    n_parts = ddf.npartitions
    sample_idx = list(range(min(_cardinality_sample_partitions, n_parts)))
    sample_pdf = dd.concat([ddf.partitions[i] for i in sample_idx])[numeric_cols].compute()

    low_card_cols = [
        c for c in numeric_cols
        if sample_pdf[c].nunique(dropna=True) < max_cardinality_for_numeric
    ]

    return dtype_cat_cols + low_card_cols


class LazyCategoricalEncoder:
    """
    Fit-once / transform-many categorical handler for Dask DataFrames that
    never materializes the full dataset. Replaces the duplicated, fully-eager
    `_encode_categorical` methods previously copy-pasted (with drifting
    bugs) across logistic.py, random_forest.py, and catboost_model.py.

    Unseen categories at transform time are mapped to a reserved "MISSING"
    bucket that is always included at fit time, so transform never raises.

    `ordinal_encode` controls the final step:
      - True (default): categories are integer-ordinal-encoded via
        `dask_ml.preprocessing.OrdinalEncoder`. Required for models with no
        native categorical support -- logistic regression (needs a purely
        numeric design matrix to scale/fit) and the blockwise-voting random
        forest (scikit-learn's RandomForestClassifier also has no native
        categorical handling).
      - False: the transform stops after `Categorizer`, i.e. columns come
        back as pandas `category` dtype rather than integer codes. This is
        what XGBoost's `enable_categorical=True` and LightGBM's
        `categorical_feature='auto'` are built to consume directly --
        passing genuine category dtype (instead of an arbitrary ordinal
        integer encoding, which invents a false ordering between
        categories) is the more memory- and accuracy-efficient native path
        for those two GBM libraries. This is why requirement #5 ("most
        memory-efficient backend for each algorithm") gets per-model
        treatment here instead of one shared numeric encoding for
        everything.
    """

    def __init__(self, categorical_columns=None, ordinal_encode: bool = True):
        self.categorical_columns = categorical_columns
        self.ordinal_encode = ordinal_encode
        self._categorizer = None
        self._encoder = None
        self._fitted = False

    def fit(self, ddf: dd.DataFrame) -> "LazyCategoricalEncoder":
        if self.categorical_columns is None:
            self.categorical_columns = identify_categorical_columns(ddf)

        if not self.categorical_columns:
            self._fitted = True
            return self

        logger.info(
            f"  Fitting lazy categorical {'encoder' if self.ordinal_encode else 'typer'} on "
            f"{len(self.categorical_columns)} columns "
            f"(no full-dataset materialization)..."
        )

        ddf_prepped = self._prep_missing(ddf)

        self._categorizer = Categorizer(columns=self.categorical_columns)
        ddf_cat = self._categorizer.fit_transform(ddf_prepped)

        if self.ordinal_encode:
            self._encoder = OrdinalEncoder(columns=self.categorical_columns)
            self._encoder.fit(ddf_cat)

        self._fitted = True
        return self

    def transform(self, ddf: dd.DataFrame) -> dd.DataFrame:
        if not self._fitted:
            raise ValueError("LazyCategoricalEncoder must be fit() before transform().")
        if not self.categorical_columns:
            return ddf

        ddf_prepped = self._prep_missing(ddf)

        # Map categories unseen at fit time onto the reserved "MISSING"
        # bucket rather than letting Categorizer introduce new codes that
        # the fitted OrdinalEncoder (or downstream native model) wouldn't
        # recognize. `Categorizer.categories_` maps column -> pandas
        # CategoricalDtype (NOT a plain list of category values) -- pull
        # `.categories` off each one to get the actual allowed-value list.
        # Explicitly ensure "MISSING" is always an allowed category (it may
        # not have been observed at fit time if that column happened to
        # have zero nulls in the training partitions), so the
        # unseen-category fallback below never silently produces NaN.
        known_categories = {}
        for col in self.categorical_columns:
            cats = list(self._categorizer.categories_[col].categories)
            if "MISSING" not in cats:
                cats = cats + ["MISSING"]
            known_categories[col] = cats

        def _restrict_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.copy()
            for col in self.categorical_columns:
                allowed = known_categories[col]
                pdf[col] = pdf[col].where(pdf[col].isin(allowed), "MISSING")
                pdf[col] = pd.Categorical(pdf[col], categories=allowed)
            return pdf

        meta = ddf_prepped._meta.copy()
        for col in self.categorical_columns:
            meta[col] = pd.Categorical([], categories=known_categories[col])

        ddf_restricted = ddf_prepped.map_partitions(_restrict_partition, meta=meta)

        if not self.ordinal_encode:
            # Native path: hand back genuine category-dtype columns as-is.
            return ddf_restricted

        ddf_encoded = self._encoder.transform(ddf_restricted)
        return ddf_encoded

    def fit_transform(self, ddf: dd.DataFrame) -> dd.DataFrame:
        return self.fit(ddf).transform(ddf)

    def _prep_missing(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Fill nulls with the literal string 'MISSING' so it becomes its own
        category, and coerce categorical columns to string dtype up front
        (partition-wise, lazy -- no compute)."""
        ddf = ddf.copy()
        for col in self.categorical_columns:
            ddf[col] = ddf[col].astype("object").fillna("MISSING").astype(str)
        return ddf
