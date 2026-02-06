"""
Genera 4 versiones del radar con diferentes configuraciones
para encontrar la orientación correcta
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

def create_test_frame_with_radar(video_path, frame_num, output_path, flip_x, flip_y, rotation, label):
    """Crea un frame de prueba con radar configurado de una manera específica"""

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

            # Crear transformer con parámetros específicos
            class CustomTransformer(ViewTransformer):
                def __init__(self, source, target, flip_x, flip_y):
                    super().__init__(source, target)
                    self.flip_x = flip_x
                    self.flip_y = flip_y

                def transform_points(self, points, flip_x=None, flip_y=None):
                    if self.m is None or points is None or points.size == 0:
                        return points

                    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
                    transformed = cv2.perspectiveTransform(reshaped, self.m)
                    transformed = transformed.reshape(-1, 2)

                    # Aplicar inversiones
                    if self.flip_x:
                        transformed[:, 0] = 105.0 - transformed[:, 0]
                    if self.flip_y:
                        transformed[:, 1] = 68.0 - transformed[:, 1]

                    return transformed

            transformer = CustomTransformer(valid_keypoints, target_points, flip_x, flip_y)

            # Transformar posiciones de jugadores
            if len(person_detections) > 0:
                player_positions = np.column_stack([
                    (person_detections.xyxy[:, 0] + person_detections.xyxy[:, 2]) / 2,
                    person_detections.xyxy[:, 3]
                ])

                transformed_positions = transformer.transform_points(player_positions)

                # Crear radar
                points_dict = {'team1': transformed_positions}
                radar_view = draw_radar_view(pitch_config, points_dict, scale=8)

                # Aplicar rotación
                if rotation == 90:
                    radar_view = cv2.rotate(radar_view, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    radar_view = cv2.rotate(radar_view, cv2.ROTATE_180)
                elif rotation == 270:
                    radar_view = cv2.rotate(radar_view, cv2.ROTATE_90_CLOCKWISE)

                # Redimensionar y colocar
                scale_factor = 0.25
                new_w = int(width * scale_factor)
                aspect_ratio = radar_view.shape[0] / radar_view.shape[1]
                new_h = int(new_w * aspect_ratio)
                radar_resized = cv2.resize(radar_view, (new_w, new_h))

                # Posición: esquina superior derecha
                margin = 20
                offset_x = width - new_w - margin
                offset_y = margin

                # Agregar etiqueta
                cv2.putText(frame, label, (offset_x, offset_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Colocar radar
                if offset_y + new_h <= height and offset_x >= 0:
                    frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = radar_resized

    # Guardar
    cv2.imwrite(str(output_path), frame)
    print(f"Guardado: {output_path.name}")


def main():
    video_path = Path("inputs/1_720p_clip_300-20.mp4")
    output_dir = Path("outputs/test_orientations")
    output_dir.mkdir(exist_ok=True)

    frame_num = 250  # Frame del medio

    print("Generando 4 versiones del radar...")
    print("="*60)

    configurations = [
        # (flip_x, flip_y, rotation, label, filename)
        (False, False, 90, "A: Sin flip, 90deg", "test_A_no_flip_90deg.jpg"),
        (True, False, 90, "B: Flip X, 90deg", "test_B_flip_x_90deg.jpg"),
        (False, True, 90, "C: Flip Y, 90deg", "test_C_flip_y_90deg.jpg"),
        (True, True, 90, "D: Flip X+Y, 90deg", "test_D_flip_xy_90deg.jpg"),
    ]

    for flip_x, flip_y, rotation, label, filename in configurations:
        output_path = output_dir / filename
        print(f"\n{label}...")
        create_test_frame_with_radar(
            video_path, frame_num, output_path,
            flip_x, flip_y, rotation, label
        )

    print("\n" + "="*60)
    print("Frames generados en: outputs/test_orientations/")
    print("\nCompara las 4 versiones:")
    print("  A: Sin inversión")
    print("  B: Invertido en X (izquierda-derecha)")
    print("  C: Invertido en Y (arriba-abajo)")
    print("  D: Invertido en X e Y (ambos)")
    print("\nDime cuál coincide mejor con las posiciones reales.")
    print("="*60)


if __name__ == "__main__":
    main()
