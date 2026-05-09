# src/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()


CHOSEN_PROVIDER = os.getenv("MODEL_PROVIDER", "groq").lower()

