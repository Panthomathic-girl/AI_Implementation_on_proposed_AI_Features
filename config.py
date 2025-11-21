# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "20"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def __post_init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required in .env")

settings = Settings()