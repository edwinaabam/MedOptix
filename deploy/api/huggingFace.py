from huggingface_hub import HfApi
from dotenv import load_dotenv
from pathlib import Path
import os

# --------------------------------------------------
# Resolve project paths (single source of truth)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIR / "sarimax_model.pkl"
SCHEMA_PATH = MODEL_DIR / "sarimax_schema.json"

print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
#load_dotenv()

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

token = os.getenv("HF_token")
if not token:
    raise ValueError("HF_token not found in .env")

# --------------------------------------------------
# Hugging Face upload
# --------------------------------------------------
api = HfApi(token=token)
print("Authenticated as:", api.whoami())

api.upload_file(
    path_or_fileobj=str(MODEL_PATH),
    path_in_repo="sarimax_model.pkl",
    repo_id="edabam2026/medoptix_admission_model",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj=str(SCHEMA_PATH),
    path_in_repo="sarimax_schema.json",
    repo_id="edabam2026/medoptix_admission_model",
    repo_type="model"
)

print("Uploading model to Hugging Face...")

