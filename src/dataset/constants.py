import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv()

IMAGES_DIR = os.getenv("IMAGES_DIR", "images")
