import numpy as np
import pyarrow.dataset as ds
import xgboost as xgb

class ParquetBatchIter(xgb.core.DataIter):
    def __init__(
        self,
        dataset: ds.Dataset,
        feature_cols,
        label_col,
        cat_cols,
        cat_maps,              # dict: col -> dict(value -> int) OR None if hashing
        batch_size=200_000,
        missing=np.nan,
    ):
        super().__init__()
        self.dataset = dataset
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.cat_cols = set(cat_cols)
        self.cat_maps = cat_maps
        self.batch_size = batch_size
        self.missing = missing

        self._scanner = None
        self._batches = None

    def reset(self):
        # Create a fresh scanner each epoch / DMatrix construction
        self._scanner = self.dataset.scanner(
            columns=self.feature_cols + [self.label_col],
            batch_size=self.batch_size,
            use_threads=True,
        )
        self._batches = iter(self._scanner.to_batches())

    def _encode_cats(self, arr, col):
        # arr is a numpy array (object/string)
        # Option A: dictionary encoding
        if self.cat_maps is not None and col in self.cat_maps:
            m = self.cat_maps[col]
            # unknown -> 0
            # vectorized-ish mapping:
            out = np.zeros(arr.shape[0], dtype=np.uint32)
            # For speed on huge batches, consider pandas.Series(arr).map(m).fillna(0)
            for i, v in enumerate(arr):
                out[i] = m.get(v, 0)
            return out

        # Option B: hashing encoding (simple fallback)
        # (Use a better hash like xxhash/mmh3 in production)
        return (np.vectorize(hash)(arr).astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)

    def next(self, input_data):
        try:
            batch = next(self._batches)
        except StopIteration:
            return 0

        tbl = batch.to_pandas(types_mapper=None)  # could keep as Arrow longer if you want
        y = tbl[self.label_col].to_numpy()

        X_parts = []
        feature_types = []
        for col in self.feature_cols:
            a = tbl[col].to_numpy()
            if col in self.cat_cols:
                a = self._encode_cats(a, col).astype(np.uint32)
                feature_types.append("c")  # categorical
            else:
                # numeric; leave missing as np.nan
                a = a.astype(np.float32, copy=False)
                feature_types.append("q")  # quantitative
            X_parts.append(a.reshape(-1, 1))

        X = np.concatenate(X_parts, axis=1)

        input_data(data=X, label=y, feature_types=feature_types)
        return 1


# ----
# dataset = ds.dataset("s3://bucket/path/", format="parquet")  # configure S3 FS as needed

it = ParquetBatchIter(
    dataset=dataset_train,
    feature_cols=feature_cols,
    label_col="label",
    cat_cols=cat_cols,
    cat_maps=cat_maps,       # or None to use hashing
    batch_size=200_000,
)

dtrain = xgb.QuantileDMatrix(
    it,
    max_bin=256,
    enable_categorical=True,
)

# Validation can also be a streaming iterator, or a smaller in-memory sample
dvalid = ...

params = dict(
    objective="binary:logistic",
    eval_metric=["aucpr", "auc", "logloss"],
    tree_method="hist",
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    # class imbalance helpers:
    scale_pos_weight=SPW,  # e.g. (neg/pos) from a scan
)

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=100,
)
# ----
import polars as pl

# categories_by_col: dict[str, list[str]]
enum_dtypes = {
    col: pl.Enum(categories)
    for col, categories in categories_by_col.items()
}

def preprocess_batch(df: pl.DataFrame) -> pl.DataFrame:
    # cast to Enum
    df = df.with_columns([
        pl.col(col).cast(enum_dtypes[col], strict=False).fill_null("__OTHER__")
        for col in enum_dtypes
    ])

    # turn Enum to its physical integer codes for XGBoost
    # (Polars represents Enum physically as integers; exact method can vary by version)
    df = df.with_columns([
        pl.col(col).to_physical().cast(pl.UInt32).alias(col)
        for col in enum_dtypes
    ])
    return df

'''
strict=False 
prevents hard failures when unseen categories show up (they’ll become null, and you can map to "__OTHER__" or 0)

to_physical() is the important step: it gives you the stable integer codes that XGBoost wants.

When Enum is a bad idea
If a column has millions of unique values, building/storing an Enum category list can get huge and slow.
If categories evolve constantly (new values daily), you’ll need a policy for unknowns (OTHER bucket or hashing).
'''

#
import polars as pl

CATS = {
    "state": ["AL","AK", "..."],
    "channel": ["email","sms","push","__OTHER__"],
    # ...
}

ENUM_DTYPES = {c: pl.Enum(v) for c, v in CATS.items()}

# ---
def encode_enums(df: pl.DataFrame, enum_dtypes: dict[str, pl.Enum]) -> pl.DataFrame:
    enum_cols = list(enum_dtypes.keys())

    # Cast strings -> Enum; unknowns become null with strict=False
    df = df.with_columns([
        pl.col(c).cast(enum_dtypes[c], strict=False)
        for c in enum_cols
    ])

    # Replace unknowns with "__OTHER__" (must exist in your category list)
    df = df.with_columns([
        pl.when(pl.col(c).is_null())
          .then(pl.lit("__OTHER__").cast(enum_dtypes[c]))
          .otherwise(pl.col(c))
          .alias(c)
        for c in enum_cols
    ])

    # Convert Enum -> physical integer codes for XGBoost
    df = df.with_columns([
        pl.col(c).to_physical().cast(pl.UInt32).alias(c)
        for c in enum_cols
    ])

    return df
