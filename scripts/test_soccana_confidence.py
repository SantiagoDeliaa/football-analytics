"""
Script para probar diferentes umbrales de confianza en Soccana
y ver cuántos keypoints detectamos
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from ultralytics import YOLO

def test_confidence_thresholds():
    print("\n" + "="*80)
    print("TEST: Umbrales de Confianza para Soccana")
    print("="*80 + "\n")

    # Paths
    model_path = project_root / "models" / "soccana_keypoint" / "Model" / "weights" / "best.pt"
    video_path = project_root / "inputs" / "2_720p_clip_5-20.mp4"

    print("Cargando modelo Soccana...")
    model = YOLO(str(model_path))

    print("Leyendo frame de prueba (frame 100)...")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: No se pudo leer el frame")
        return False

    print("Detectando keypoints...\n")
    results = model(frame, verbose=False, conf=0.1)  # Umbral muy bajo para inferencia
    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        print("ERROR: No se detectaron keypoints")
        return False

    keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
    keypoints_conf = result.keypoints.conf.cpu().numpy()[0]

    print("="*80)
    print("ANALISIS DE CONFIANZAS")
    print("="*80 + "\n")

    # Probar diferentes umbrales
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    print(f"{'Umbral':<10} {'Keypoints':<12} {'IDs Detectados'}")
    print("-" * 80)

    best_threshold = None
    best_count = 0

    for thresh in thresholds:
        valid_mask = keypoints_conf >= thresh
        valid_count = valid_mask.sum()
        valid_ids = np.where(valid_mask)[0].tolist()

        print(f"{thresh:<10.2f} {valid_count:<12} {valid_ids}")

        # Buscar umbral óptimo (al menos 8 keypoints distribuidos)
        if 8 <= valid_count <= 15 and valid_count > best_count:
            best_count = valid_count
            best_threshold = thresh

    print("\n" + "="*80)
    print("KEYPOINTS POR ZONA (Umbral 0.15)")
    print("="*80 + "\n")

    valid_mask = keypoints_conf >= 0.15
    valid_ids = np.where(valid_mask)[0]

    # Categorizar keypoints por zona
    corners = [0, 16, 9, 25]
    center = [10, 11, 12, 13, 14, 15, 26, 27, 28]
    penalty_left = [1, 2, 3, 4]
    penalty_right = [17, 18, 19, 20]
    goal_left = [5, 6, 7, 8]
    goal_right = [21, 22, 23, 24]

    zones = {
        'Esquinas': [id for id in valid_ids if id in corners],
        'Centro/Círculo': [id for id in valid_ids if id in center],
        'Área Penal Izq': [id for id in valid_ids if id in penalty_left],
        'Área Penal Der': [id for id in valid_ids if id in penalty_right],
        'Área Pequeña Izq': [id for id in valid_ids if id in goal_left],
        'Área Pequeña Der': [id for id in valid_ids if id in goal_right],
    }

    for zone, ids in zones.items():
        if ids:
            confs = [keypoints_conf[i] for i in ids]
            print(f"{zone:<20} {len(ids)} keypoints: {ids}")
            print(f"{'':20} Confianzas: {[f'{c:.3f}' for c in confs]}")
        else:
            print(f"{zone:<20} 0 keypoints")

    print("\n" + "="*80)
    print("RECOMENDACION")
    print("="*80 + "\n")

    if best_threshold:
        print(f"Umbral recomendado: {best_threshold}")
        print(f"  - Detecta {best_count} keypoints")
        print(f"  - Balance entre cobertura y precisión")
    else:
        print("Umbral actual (0.3) es adecuado para este video")
        print("  - Considera usar aproximación de campo completo")

    print("\nNOTA: Este video muestra principalmente el círculo central.")
    print("      Para homografía robusta necesitamos:")
    print("      - Al menos 4 keypoints distribuidos por el campo")
    print("      - Idealmente de diferentes zonas (no solo círculo central)")

    return True

if __name__ == "__main__":
    success = test_confidence_thresholds()
    sys.exit(0 if success else 1)
