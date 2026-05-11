"""Load runtime configuration from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()


CHOSEN_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()

