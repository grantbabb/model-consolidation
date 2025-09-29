import argparse
import os
import time
import tempfile
from urllib.parse import urlparse

import boto3
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-group-name", type=str, required=True)
    parser.add_argument("--output-train-dir", type=str, required=True)
    parser.add_argument("--output-validation-dir", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--event-time-after", type=str, default="")
    parser.add_argument("--event-time-column", type=str, default="event_time")
    parser.add_argument("--train-split-ratio", type=float, default=0.8)
    parser.add_argument("--athena-output-s3", type=str, required=True)
    return parser.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _s3_join(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p is not None]
    return "/".join(cleaned)


def _wait_for_athena_query(athena, query_execution_id: str) -> dict:
    while True:
        resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return resp
        time.sleep(2)


def _download_s3_object(s3, s3_uri: str, local_path: str) -> None:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    _ensure_dir(os.path.dirname(local_path))
    s3.download_file(bucket, key, local_path)


def main() -> None:
    args = parse_args()

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS region not found in environment.")

    sm = boto3.client("sagemaker", region_name=region)
    athena = boto3.client("athena", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    # Discover offline store catalog location from the Feature Group
    fg_desc = sm.describe_feature_group(FeatureGroupName=args.feature_group_name)
    offline_cfg = fg_desc.get("OfflineStoreConfig") or {}
    data_catalog = offline_cfg.get("DataCatalogConfig") or {}

    glue_database = data_catalog.get("Database")
    glue_table = data_catalog.get("TableName")

    if not glue_database or not glue_table:
        raise RuntimeError(
            "Offline store DataCatalogConfig (Database/TableName) not found for the Feature Group."
        )

    # Athena output location provided explicitly
    athena_results_s3 = args.athena_output_s3

    # Compose query with optional time filter
    select_sql = f"SELECT * FROM \"{glue_database}\".\"{glue_table}\""
    if args.event_time_after:
        select_sql += (
            f" WHERE {args.event_time_column} > timestamp '{args.event_time_after}'"
        )

    query_resp = athena.start_query_execution(
        QueryString=select_sql,
        ResultConfiguration={"OutputLocation": athena_results_s3},
        WorkGroup="primary",
    )

    qid = query_resp["QueryExecutionId"]
    exec_desc = _wait_for_athena_query(athena, qid)
    state = exec_desc["QueryExecution"]["Status"]["State"]
    if state != "SUCCEEDED":
        reason = exec_desc["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Athena query failed with state {state}: {reason}")

    # Athena places the CSV at s3://.../<QueryExecutionId>.csv
    result_s3_uri = f"{athena_results_s3}{qid}.csv"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_csv = os.path.join(tmpdir, "athena_result.csv")
        _download_s3_object(s3, result_s3_uri, local_csv)

        df = pd.read_csv(local_csv)

        # Basic shuffle
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Split train/validation while preserving target column
        split_index = int(len(df) * float(args.train_split_ratio))
        train_df = df.iloc[:split_index]
        validation_df = df.iloc[split_index:]

        # Ensure target column exists
        if args.target_column not in df.columns:
            raise RuntimeError(
                f"Target column '{args.target_column}' not found in dataset columns: {list(df.columns)}"
            )

        # Save to the processing output directories expected by the ProcessingStep (local paths)
        _ensure_dir(args.output_train_dir)
        _ensure_dir(args.output_validation_dir)

        train_out = os.path.join(args.output_train_dir, "train.csv")
        val_out = os.path.join(args.output_validation_dir, "validation.csv")

        train_df.to_csv(train_out, index=False, header=True)
        validation_df.to_csv(val_out, index=False, header=True)


if __name__ == "__main__":
    main()

