"""Shared configuration for example agents."""

import os
from pathlib import Path

ANTHROPIC_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
TRACES_DIR = Path("./traces")
MAX_TOKENS_PER_CALL = 1024
MAX_ACTIONS_PER_SESSION = 50
