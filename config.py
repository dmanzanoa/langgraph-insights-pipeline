"""
Configuration and AWS client setup for the LangGraph insights pipeline.

This module centralises all configuration related to AWS services and model
identifiers.  Sensitive values such as the AWS region, S3 buckets and
Bedrock model identifiers are read from environment variables.  Default
placeholders are provided so that the code runs in development without
exposing secrets.  When deploying to production, set the appropriate
environment variables to match your infrastructure.
"""

import os
import boto3
from botocore.config import Config


# -----------------------------------------------------------------------------
# Basic configuration
#
# Values below fall back to sensible defaults if the corresponding
# environment variable is not defined.  You should override these in your
# deployment environment to point at your own resources.
# -----------------------------------------------------------------------------

# The AWS region used for all service clients.  Defaults to us‑east‑1.
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Input and output S3 buckets.  The input bucket is where parquet files
# containing raw conversations are stored.  The output bucket is used to
# persist generated insights and tendencias.  Both default to the same
# placeholder bucket name.
S3_BUCKET = os.getenv("S3_BUCKET", "eess-silver-layer")
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", S3_BUCKET)

# Identifiers for the Anthropic models hosted on Bedrock.  These should be
# replaced with your own model versions if you have different deployments.
MODEL_MAIN = os.getenv("MODEL_MAIN", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
MODEL_COMPRESS = os.getenv("MODEL_COMPRESS", "global.anthropic.claude-haiku-4-5-20251001-v1:0")

# Optional: override the maximum number of retries for LLM calls.  A small
# number helps avoid runaway loops in case of persistent failure.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# Data sources mapped by label.  Each entry corresponds to an S3 prefix
# containing parquet files for a particular pipeline run (e.g. subsidio,
# no_subsidio, recomendador).  You can override the prefixes via
# environment variables if your data layout differs.
DATA_SOURCES = {
    "subsidio": os.getenv("DATA_SOURCE_SUBSIDIO", "comercial/Lidz_mensajeria/subsidio/"),
    "no_subsidio": os.getenv("DATA_SOURCE_NO_SUBSIDIO", "comercial/Lidz_mensajeria/no_subsidio/"),
    "recomendador": os.getenv("DATA_SOURCE_RECOMENDADOR", "comercial/Lidz_mensajeria/recomendador"),
}

# -----------------------------------------------------------------------------
# AWS clients
#
# To tune concurrency and retry behaviour, adjust the values in the Config
# object.  They are chosen conservatively here to ensure stability when
# calling Bedrock and S3.
# -----------------------------------------------------------------------------

# Shared boto3 session per module.  This avoids re‑creating clients on every
# call and centralises configuration in one place.
_boto_config = Config(
    read_timeout=1000,
    retries={"max_attempts": 3, "mode": "standard"},
    max_pool_connections=10,
)

_session = boto3.Session(region_name=AWS_REGION)

# Lazily initialised clients for S3, Bedrock Runtime and CloudWatch.  They
# reference the same boto3 session and share the connection pool defined
# above.  See boto3 documentation for further details.
s3 = _session.client("s3", config=_boto_config)
bedrock = _session.client("bedrock-runtime", config=_boto_config)
cloudwatch = _session.client("cloudwatch", config=_boto_config)

__all__ = [
    "AWS_REGION",
    "S3_BUCKET",
    "OUTPUT_BUCKET",
    "MODEL_MAIN",
    "MODEL_COMPRESS",
    "MAX_RETRIES",
    "DATA_SOURCES",
    "s3",
    "bedrock",
    "cloudwatch",
]