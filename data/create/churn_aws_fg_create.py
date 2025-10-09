import boto3
import awswrangler as wr
import pandas as pd
from datetime import datetime, timezone
from pandas.api.types import is_integer_dtype, is_float_dtype, is_bool_dtype
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REGION = "us-west-2"
BUCKET = "datasets-in-out"
PREFIX = "feature-store/offline/churn_aws"
ROLE_ARN = "arn:aws:iam::063299843915:role/service-role/AmazonSageMaker-ExecutionRole-20250522T112887"
FEATURE_GROUP_NAME = "churn_aws_fg"
S3_INPUT_PATH = "input/churn-aws/full_feature_data.csv"

session = boto3.Session(region_name=REGION)
sm = session.client("sagemaker")

in_path = f"s3://{BUCKET}/{S3_INPUT_PATH}"
df = wr.s3.read_csv(in_path)

# Ensure required event time column exists (string ISO 8601 recommended)
if "event_time" not in df.columns:
    df["event_time"] = pd.Timestamp.now(tz="UTC").isoformat()

def infer_feature_definitions(frame: pd.DataFrame):
    feature_defs = []
    for col in frame.columns:
        series = frame[col]
        if is_integer_dtype(series) or is_bool_dtype(series):
            ftype = "Integral"
        elif is_float_dtype(series):
            ftype = "Fractional"
        else:
            ftype = "String"
        feature_defs.append({"FeatureName": col, "FeatureType": ftype})
    return feature_defs


def row_to_record(row: pd.Series):
    record = []
    for feature_name, value in row.items():
        if pd.isna(value):
            continue
        record.append({"FeatureName": str(feature_name), "ValueAsString": str(value)})
    return record


def put_row(row: pd.Series):
    return client.put_record(
        FeatureGroupName=fg_name,
        Record=row_to_record(row),
        # TargetStores=["OnlineStore", "OfflineStore"],  # uncomment if you want both
    )


feature_definitions = infer_feature_definitions(df)
offline_store_s3_uri = f"s3://{BUCKET}/{PREFIX}/"
fg_name = f"{FEATURE_GROUP_NAME}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

# IMPORTANT: Ensure "userId" exists in df
record_identifier = "userId"
if record_identifier not in df.columns:
    raise ValueError(f"Record identifier column '{record_identifier}' not found in data.")

sm.create_feature_group(
    FeatureGroupName=fg_name,
    RecordIdentifierFeatureName=record_identifier,
    EventTimeFeatureName="event_time",
    FeatureDefinitions=feature_definitions,
    OnlineStoreConfig={"EnableOnlineStore": True},
    OfflineStoreConfig={"S3StorageConfig": {"S3Uri": offline_store_s3_uri}},
    RoleArn=ROLE_ARN,
    Description="Created via boto3",
)

# Wait for creation to complete
while True:
    resp = sm.describe_feature_group(FeatureGroupName=fg_name)
    status = resp["FeatureGroupStatus"]
    print("FeatureGroupStatus:", status)
    if status in ("Created", "CreateFailed"):
        break
    time.sleep(5)

print("Feature group name:", fg_name, "status:", status)

client = boto3.client("sagemaker-featurestore-runtime", region_name=REGION)

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(put_row, row) for _, row in df.iterrows()]
    for f in as_completed(futures):
        f.result()  # raises if any put_record failed

