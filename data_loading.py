"""Data loading helpers for the insights pipeline."""

import io
import pandas as pd
import pyarrow.parquet as pq
from . import config


def load_parquet_folder(prefix):
    """
    Load all Parquet files from a prefix in the input S3 bucket and
    concatenate them into a single DataFrame.

    The S3 client configured in :mod:`config` is used to list objects
    recursively and download each Parquet file.  Files are concatenated
    into a single DataFrame.  If no files are found an empty DataFrame is
    returned.

    Args:
        prefix: The key prefix within the bucket where Parquet files live.

    Returns:
        A concatenated DataFrame containing the data from all Parquet files
        under the prefix.
    """
    s3 = config.s3
    bucket = config.S3_BUCKET
    dfs = []
    continuation_token = None
    while True:
        if continuation_token:
            resp = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=continuation_token,
            )
        else:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                raw = s3.get_object(Bucket=bucket, Key=key)
                table = pq.read_table(io.BytesIO(raw["Body"].read()))
                df = table.to_pandas()
                dfs.append(df)
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
