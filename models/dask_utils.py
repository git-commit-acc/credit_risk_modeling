# models/dask_utils.py
"""
Shared Dask infrastructure for the modeling layer.

This module exists to fix two systemic problems found across the original
models/*.py files:

  1. EVERY model (logistic, random_forest, xgboost, lightgbm) created its own
     `dask.distributed.Client(n_workers=3, threads_per_worker=4)` inside
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

import tempfile
import time
import os
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


os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"  # Low value encourages trimming

logger = logging.getLogger(__name__)

# Global client instance
_shared_client: Optional[Client] = None
_shared_cluster: Optional[LocalCluster] = None


def trim_dask_workers_memory(client: Client) -> int:
    """
    Manually trim memory on all Dask workers (Linux only).

    This function runs malloc_trim on each worker, which forces the glibc
    memory allocator to release freed memory back to the OS.

    Args:
        client: The Dask client connected to the workers.

    Returns:
        The number of bytes released.
    """
    try:
        import ctypes

        def trim_memory() -> int:
            """Call malloc_trim to release memory back to the OS."""
            try:
                libc = ctypes.CDLL("libc.so.6")
                return libc.malloc_trim(0)
            except OSError:
                return 0

        results = client.run(trim_memory)
        total_freed = sum(results.values())
        logger.info(f"Manually trimmed memory on workers: freed {total_freed / 1024 / 1024:.2f} MB")
        return total_freed
    except Exception as e:
        logger.warning(f"Could not manually trim worker memory: {e}")
        return 0


# def get_dask_client(
#     n_workers: int = 4,
#     threads_per_worker: int = 2,
#     memory_limit: str = "5GB",
#     dashboard_address: Optional[str] = None,
# ) -> Client:
#     """
#     Return the process-wide Dask distributed Client, creating it on first
#     call. Every model module should call this instead of constructing its
#     own Client -- this is what makes it safe for main.py to train five base
#     models, a meta-learner, and run hyperparameter tuning without spawning a
#     new local cluster on every `fit()` call.
#     """
#     global _client
#     with _client_lock:
#         if _client is not None:
#             try:
#                 # Cheap liveness check; if the scheduler died, fall through
#                 # and recreate rather than handing back a dead client.
#                 if _client.status == "running":
#                     # Test with a lightweight operation
#                     _client.run(lambda: 1)  # Quick heartbeat check
#                     return _client
#             except Exception:
#                 pass
#          # Use a stable temp directory for Windows
#         temp_dir = os.path.join(tempfile.gettempdir(), "dask_credit_risk")
#         os.makedirs(temp_dir, exist_ok=True)

#         logger.info(
#             f"Starting shared Dask LocalCluster "
#             f"(workers={n_workers}, threads_per_worker={threads_per_worker}, "
#             f"memory_limit={memory_limit})..."
#         )
#         cluster = LocalCluster(
#             n_workers=n_workers,
#             threads_per_worker=threads_per_worker,
#             memory_limit=memory_limit,
#             dashboard_address=dashboard_address,
#             processes=True,
#             # Add Windows-specific settings
#             local_directory=temp_dir,
#             # Shutdown timeout to allow proper cleanup
#             death_timeout=30,
#         )
#         _client = Client(cluster)
#         logger.info(f"Dask dashboard: {_client.dashboard_link}")
#         return _client

def get_dask_client(
    n_workers: int = 2,
    threads_per_worker: int = 1,
    memory_limit: str = "4GB",
    dashboard_address: str = ":8787",
    local_directory: Optional[str] = None
) -> Client:
    """
    Get or create a shared Dask client.

    Args:
        n_workers: Number of Dask workers
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        dashboard_address: Dashboard address
        local_directory: Directory for worker spilling

    Returns:
        Dask Client instance
    """
    global _shared_client, _shared_cluster

    if _shared_client is not None and _shared_client.status == 'running':
        logger.info("Reusing existing Dask client")
        return _shared_client

    # Clean up old temp directories
    _cleanup_dask_temp()

    # Create a clean temp directory for this session
    if local_directory is None:
        temp_dir = os.path.join(tempfile.gettempdir(), "dask_scratch_space")
        local_directory = temp_dir

    os.makedirs(local_directory, exist_ok=True)

    logger.info(f"Creating Dask client with {n_workers} workers...")

    try:
        # Create cluster with Windows-friendly settings
        _shared_cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=dashboard_address,
            local_directory=local_directory,
            silence_logs=logging.ERROR,
            death_timeout=60
            # Use 'managed' memory measure to avoid overreaction to unmanaged memory
            # memory_target_fraction=0.6,  # Start spilling at 60% of managed memory
            # memory_spill_fraction=0.7,   # More aggressive spilling at 70% of process memory
            # memory_pause_fraction=0.8,   # Pause at 80% of process memory
            # memory_terminate_fraction=0.95  # Terminate at 95% of process memory
        )

        _shared_client = Client(_shared_cluster)
        logger.info(f"Dask client created: {_shared_client.dashboard_link}")

        # --- Attempt to trim memory after some tasks ---
        # This is a proactive measure. You can also call this function
        # later in your pipeline if needed.
        # trim_dask_workers_memory(_shared_client)

        return _shared_client

    except Exception as e:
        logger.warning(f"Failed to create Dask cluster: {e}")
        # Fallback: Single-threaded mode
        logger.info("Falling back to single-threaded mode...")
        _shared_client = Client(processes=False, threads_per_worker=1)
        return _shared_client



# def close_dask_client() -> None:
#     """Shut down the shared client/cluster. Call once at pipeline teardown."""
#     global _client
#     with _client_lock:
#         if _client is not None:
#             try:
#                 _client.close()
#                 logger.info("Shared Dask client closed.")
#             except Exception as e:
#                 logger.warning(f"Error closing Dask client: {e}")
#             finally:
#                 _client = None

# def close_dask_client() -> None:
#     """Shut down the shared client/cluster with Windows-friendly cleanup."""
#     global _client
#     with _client_lock:
#         if _client is not None:
#             try:
#                 # Graceful shutdown - give workers time to clean up
#                 _client.shutdown()
#                 _client.close()
#                 logger.info("Shared Dask client closed gracefully.")
#             except Exception as e:
#                 logger.warning(f"Error closing Dask client: {e}")
#             finally:
#                 _client = None
    
#     # Additional Windows temp cleanup
#     import shutil
#     import tempfile
#     try:
#         temp_dir = os.path.join(tempfile.gettempdir(), "dask-scratch-space")
#         if os.path.exists(temp_dir):
#             # Only remove if it's empty or old
#             for item in os.listdir(temp_dir):
#                 item_path = os.path.join(temp_dir, item)
#                 try:
#                     if os.path.isdir(item_path):
#                         # Check if it's a dask worker directory
#                         if item.startswith("worker-") or item.startswith("scheduler-"):
#                             # Check if the directory is empty before removing
#                             if not os.listdir(item_path):
#                                 os.rmdir(item_path)
#                 except Exception:
#                     pass
#     except Exception:
#         pass

def close_dask_client():
    """Close the shared Dask client and cluster."""
    global _shared_client, _shared_cluster

    if _shared_client is not None:
        try:
            _shared_client.close()
            logger.info("Dask client closed")
        except Exception as e:
            logger.warning(f"Error closing Dask client: {e}")
        _shared_client = None

    if _shared_cluster is not None:
        try:
            _shared_cluster.close()
            logger.info("Dask cluster closed")
        except Exception as e:
            logger.warning(f"Error closing Dask cluster: {e}")
        _shared_cluster = None

    # Clean up temp directories
    _cleanup_dask_temp()


def _cleanup_dask_temp():
    """Clean up Dask temporary directories."""
    temp_dir = os.path.join(tempfile.gettempdir(), "dask-scratch-space")
    try:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up {temp_dir}")
    except Exception:
        pass


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

def with_retry(func, max_retries=3, delay=5):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

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
