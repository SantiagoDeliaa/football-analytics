"""
Test del nuevo radar: horizontal, centrado y con transparencia
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from src.utils.radar import SoccerPitchConfiguration, draw_radar_view
from src.utils.view_transformer import ViewTransformer

def create_test_frame(video_path, frame_num, output_path):
    """Crea un frame de prueba con el nuevo radar transparente y centrado"""

    # Cargar modelos
    player_model = YOLO("yolov8m.pt")
    pitch_model = YOLO("models/homography.pt")

    # Leer frame específico
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"No se pudo leer el frame {frame_num}")
        return

    height, width = frame.shape[:2]

    # Detectar jugadores
    results = player_model.predict(frame, conf=0.3, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    person_detections = detections[detections.class_id == 0]

    # Detectar keypoints del campo
    pitch_results = pitch_model(frame, conf=0.01, verbose=False)[0]

    if pitch_results.keypoints is not None and len(pitch_results.keypoints) > 0:
        keypoints_xy = pitch_results.keypoints.xy.cpu().numpy()[0]
        keypoints_conf = pitch_results.keypoints.conf.cpu().numpy()[0]

        valid_mask = keypoints_conf > 0.5
        valid_keypoints = keypoints_xy[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_keypoints) >= 4:
            # Crear transformación
            pitch_config = SoccerPitchConfiguration(model_type='roboflow')
            target_points = pitch_config.get_keypoints_from_ids(valid_indices)

            transformer = ViewTransformer(valid_keypoints, target_points)

            # Transformar posiciones de jugadores
            if len(person_detections) > 0:
                player_positions = np.column_stack([
                    (person_detections.xyxy[:, 0] + person_detections.xyxy[:, 2]) / 2,
                    person_detections.xyxy[:, 3]
                ])

                # Sin flip para probar
                transformed_positions = transformer.transform_points(player_positions, flip_x=False)

                # Crear radar
                points_dict = {'team1': transformed_positions}
                radar_view = draw_radar_view(pitch_config, points_dict, scale=8)

                # Horizontal (sin rotación)
                # Tamaño: 30% del ancho
                scale_factor = 0.30
                new_w = int(width * scale_factor)
                aspect_ratio = radar_view.shape[0] / radar_view.shape[1]
                new_h = int(new_w * aspect_ratio)
                radar_resized = cv2.resize(radar_view, (new_w, new_h))

                # Centrado
                offset_x = (width - new_w) // 2
                offset_y = (height - new_h) // 2

                # Aplicar transparencia
                if offset_y + new_h <= height and offset_x >= 0:
                    alpha = 0.65  # 65% radar, 35% video
                    roi = frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
                    blended = cv2.addWeighted(roi, 1-alpha, radar_resized, alpha, 0)
                    frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = blended

    # Guardar
    cv2.imwrite(str(output_path), frame)
    print(f"Guardado: {output_path}")
    print(f"Radar: horizontal, centrado, transparencia 65%")


if __name__ == "__main__":
    video_path = Path("inputs/clip_5_25.mp4")
    output_path = Path("outputs/test_transparent_centered.jpg")
    output_path.parent.mkdir(exist_ok=True)

    print("Generando frame de prueba con radar transparente centrado...")
    create_test_frame(video_path, 250, output_path)
    print(f"\nFrame generado en: {output_path}")
