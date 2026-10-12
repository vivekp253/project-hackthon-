
# ==========================================
# GEMINI AI CLIENT
# ==========================================

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY missing in .env file"
        )

    return genai.Client(
        api_key=api_key,
        vertexai=False
    )
