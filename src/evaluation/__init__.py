"""RAGAS-backed response evaluation with analytics disabled by default."""

import os

# This executes before any submodule imports RAGAS, including in direct local runs.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
