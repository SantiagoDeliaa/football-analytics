from functools import lru_cache
from ultralytics import YOLO
from pathlib import Path
import os

# Eliminamos todas las referencias a @st.cache_resource

def load_roboflow_model(model_type: str = "yolov8m"):
    """Carga modelos YOLO para detección de jugadores."""
    # ... (mantené el resto de la lógica interna de la función)
    model_name = AVAILABLE_MODELS.get(model_type, "yolov8m")
    return YOLO(model_name)

@lru_cache(maxsize=1)
def load_model(name: str):
    """Carga y cachea el modelo YOLO indicado por nombre."""
    return YOLO(name)