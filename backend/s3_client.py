"""S3 access for the test bed (credentials from environment)."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
import pandas as pd
from botocore.exceptions import ClientError


def _credentials() -> dict:
    access = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    if not access or not secret:
        raise EnvironmentError(
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and optionally AWS_DEFAULT_REGION)."
        )
    return {
        "aws_access_key_id": access,
        "aws_secret_access_key": secret,
        "region_name": region,
    }


class TestbedS3Client:
    """Thin S3 wrapper aligned with data_pipeline.s3_sdk but env-based auth."""

    def __init__(self, bucket_name: str):
        creds = _credentials()
        self.bucket_name = bucket_name
        self.client = boto3.client("s3", **creds)
        self.resource = boto3.resource("s3", **creds)

    def list_keys(self, prefix: str, suffix: Optional[str] = None) -> List[str]:
        keys: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix is None or key.endswith(suffix):
                    keys.append(key)
        return keys

    def get_file_df(self, key: str, bucket_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        bucket = bucket_name or self.bucket_name
        print(f"[FILE READ] S3 CSV: s3://{bucket}/{key}")
        obj = self.client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=False)
        df.columns = df.columns.str.strip().str.lower()
        print(f"[FILE READ] S3 CSV done: s3://{bucket}/{key} ({len(df)} rows, {len(df.columns)} cols)")
        return df, key

    def download_file(self, key: str, local_path: str, bucket_name: Optional[str] = None) -> str:
        bucket = bucket_name or self.bucket_name
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FILE READ] S3 download: s3://{bucket}/{key} -> {path}")
        self.client.download_file(bucket, key, str(path))
        print(f"[FILE READ] S3 download done: {path}")
        return str(path)

    def upload_df(self, df: pd.DataFrame, key: str, bucket_name: Optional[str] = None) -> None:
        bucket = bucket_name or self.bucket_name
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self.resource.Object(bucket, key).put(Body=buf.getvalue())
