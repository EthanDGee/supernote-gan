import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def _load_int_tuple(variable_name: str, default: str):
    tuple_str = os.getenv(variable_name, default)
    return tuple(int(x) for x in tuple_str.strip("() ").split(","))


# IMAGES INFO
IMAGES_DIR = os.getenv("IMAGES_DIR", "converted")
NOTE_SIZE = _load_int_tuple("NOTE_SIZE", "(351, 468)")

# TRAINING INFO
DEVICE = os.getenv("DEVICE", "auto")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCH = int(os.getenv("NUM_EPOCH", "100"))
LEARNING_DISCRIM = float(os.getenv("LEARNING_DISCRIM", "0.001"))
LEARNING_GENER = float(os.getenv("LEARNING_GENER", "0.001"))
LATENT_SIZE = int(os.getenv("LATENT_SIZE", "100"))

# SAVE INFO
SAVE_DIR = os.getenv("SAVE_DIR", "checkpoints")
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "5"))
