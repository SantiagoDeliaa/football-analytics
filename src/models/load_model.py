from functools import lru_cache
from ultralytics import YOLO

@lru_cache(maxsize=1)
def load_model(name: str):
    return YOLO(name)