"""
Script simple para probar el modelo Soccana_Keypoint con un video
"""

import sys
from pathlib import Path

# Agregar directorio raiz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from ultralytics import YOLO

def test_soccana_model():
    print("\n" + "="*70)
    print("PRUEBA DE MODELO: Soccana_Keypoint (YOLOv11)")
    print("="*70 + "\n")

    # Paths
    model_path = project_root / "models" / "soccana_keypoint" / "Model" / "weights" / "best.pt"
    video_path = project_root / "inputs" / "2_720p_clip_5-20.mp4"
    output_dir = project_root / "outputs" / "soccana_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo: {model_path}")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}\n")

    # Verificar archivos
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return False

    if not video_path.exists():
        print(f"[ERROR] Video no encontrado: {video_path}")
        return False

    try:
        # Cargar modelo
        print("[1/4] Cargando modelo YOLOv11...")
        model = YOLO(str(model_path))
        print(f"[OK] Modelo cargado")
        print(f"      Tipo: {model.task}")
        print(f"      Nombres: {model.names if hasattr(model, 'names') else 'N/A'}\n")

        # Abrir video
        print("[2/4] Abriendo video...")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("[ERROR] No se pudo abrir el video")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[OK] Video abierto:")
        print(f"      Resolucion: {width}x{height}")
        print(f"      FPS: {fps}")
        print(f"      Frames totales: {total_frames}\n")

        # Procesar frame 30 (1 segundo)
        print("[3/4] Procesando frame de prueba (frame 30)...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] No se pudo leer el frame")
            cap.release()
            return False

        # Detectar keypoints
        results = model(frame, verbose=False)

        if len(results) == 0:
            print("[ADVERTENCIA] No se detectaron resultados")
            cap.release()
            return False

        result = results[0]

        # Verificar si hay keypoints
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_data = result.keypoints
            print(f"[OK] Keypoints detectados!")
            print(f"      Forma: {keypoints_data.shape if hasattr(keypoints_data, 'shape') else 'N/A'}")

            # Acceder a coordenadas
            if hasattr(keypoints_data, 'xy'):
                kps_xy = keypoints_data.xy.cpu().numpy()
                print(f"      Coordenadas XY: {kps_xy.shape}")
                print(f"      Primer keypoint: {kps_xy[0][0] if len(kps_xy) > 0 and len(kps_xy[0]) > 0 else 'N/A'}")

            if hasattr(keypoints_data, 'conf'):
                kps_conf = keypoints_data.conf.cpu().numpy()
                print(f"      Confianzas: {kps_conf.shape}")
                print(f"      Confianza promedio: {kps_conf.mean():.3f}")

        else:
            print("[ADVERTENCIA] No se encontraron keypoints en el resultado")
            print(f"      Atributos del resultado: {dir(result)}")

        # Guardar visualizacion
        print("\n[4/4] Guardando visualizacion...")

        # Usar el metodo plot() de ultralytics
        annotated_frame = result.plot()

        output_path = output_dir / "frame_30_keypoints.jpg"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"[OK] Visualizacion guardada: {output_path}\n")

        # Informacion adicional
        print("="*70)
        print("RESULTADO DEL TEST")
        print("="*70 + "\n")

        has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
        if has_keypoints and hasattr(result.keypoints, 'xy'):
            kps_xy = result.keypoints.xy.cpu().numpy()
            num_kps = len(kps_xy[0]) if len(kps_xy) > 0 else 0

            print(f"[OK] Modelo funciona correctamente!")
            print(f"      Keypoints detectados: {num_kps}")
            print(f"      Visualizacion guardada en: {output_dir}")
            print("\nProximo paso:")
            print("  1. Revisar la imagen generada para verificar calidad")
            print("  2. Si se ve bien, integrar en app.py")
            print(f"  3. Usar modelo: models/soccana_keypoint/Model/weights/best.pt\n")
            return True
        else:
            print("[ADVERTENCIA] Modelo cargado pero no detecto keypoints")
            print("              Puede ser que el frame no tenga campo visible")
            print("              o que el modelo use otro formato de salida\n")
            return False

        cap.release()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_soccana_model()
    sys.exit(0 if success else 1)
