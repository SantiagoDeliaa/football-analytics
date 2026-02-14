"""
Compara modelos de homografía: homography.pt (32 kps) vs Soccana (29 kps)
Para determinar cuál funciona mejor en tus videos
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_model_on_video(video_path, model_path, model_name, num_frames=10):
    """Prueba un modelo de keypoints en un video"""
    print(f"\n{'='*60}")
    print(f"Probando: {model_name}")
    print(f"Modelo: {model_path}")
    print(f"{'='*60}")

    # Cargar modelo
    model = YOLO(model_path)

    # Verificar keypoints
    test_result = model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    num_kps = test_result[0].keypoints.data.shape[1]
    print(f"Keypoints totales: {num_kps}")

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stats = {
        'frames_tested': 0,
        'keypoints_detected': [],
        'high_conf_keypoints': [],  # conf > 0.5
        'medium_conf_keypoints': [], # conf > 0.3
        'low_conf_keypoints': [],   # conf > 0.05
        'homography_viable': 0  # frames con >= 4 keypoints bien distribuidos
    }

    # Probar cada N frames
    step = max(1, total_frames // num_frames)

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        stats['frames_tested'] += 1

        # Detectar keypoints
        results = model(frame, conf=0.05, verbose=False)

        if results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            stats['keypoints_detected'].append(0)
            stats['high_conf_keypoints'].append(0)
            stats['medium_conf_keypoints'].append(0)
            stats['low_conf_keypoints'].append(0)
            continue

        keypoints = results[0].keypoints.data[0]  # [num_kps, 3]
        keypoints_xy = keypoints[:, :2].cpu().numpy()
        keypoints_conf = keypoints[:, 2].cpu().numpy()

        # Contar por nivel de confianza
        high_conf = keypoints_conf > 0.5
        medium_conf = keypoints_conf > 0.3
        low_conf = keypoints_conf > 0.05

        num_high = high_conf.sum()
        num_medium = medium_conf.sum()
        num_low = low_conf.sum()

        stats['keypoints_detected'].append(int(num_low))
        stats['high_conf_keypoints'].append(int(num_high))
        stats['medium_conf_keypoints'].append(int(num_medium))
        stats['low_conf_keypoints'].append(int(num_low))

        # Verificar si es viable para homografía (>=4 keypoints bien distribuidos)
        if num_low >= 4:
            valid_kps = keypoints_xy[low_conf]
            x_range = valid_kps[:, 0].max() - valid_kps[:, 0].min()
            y_range = valid_kps[:, 1].max() - valid_kps[:, 1].min()

            # Requiere distribución mínima (30% del frame)
            min_spread = frame.shape[1] * 0.3
            if x_range > min_spread or y_range > min_spread:
                stats['homography_viable'] += 1

    cap.release()

    # Calcular promedios
    avg_detected = np.mean(stats['keypoints_detected']) if stats['keypoints_detected'] else 0
    avg_high = np.mean(stats['high_conf_keypoints']) if stats['high_conf_keypoints'] else 0
    avg_medium = np.mean(stats['medium_conf_keypoints']) if stats['medium_conf_keypoints'] else 0
    avg_low = np.mean(stats['low_conf_keypoints']) if stats['low_conf_keypoints'] else 0

    homography_success_rate = (stats['homography_viable'] / stats['frames_tested']) * 100 if stats['frames_tested'] > 0 else 0

    print(f"\n📊 Resultados ({stats['frames_tested']} frames testeados):")
    print(f"   Keypoints promedio detectados:")
    print(f"     • Alta confianza (>0.5):    {avg_high:.1f}")
    print(f"     • Media confianza (>0.3):   {avg_medium:.1f}")
    print(f"     • Baja confianza (>0.05):   {avg_low:.1f}")
    print(f"   Homografía viable: {stats['homography_viable']}/{stats['frames_tested']} frames ({homography_success_rate:.1f}%)")

    return {
        'model_name': model_name,
        'num_keypoints': num_kps,
        'avg_detected_low': avg_low,
        'avg_detected_medium': avg_medium,
        'avg_detected_high': avg_high,
        'homography_success_rate': homography_success_rate,
        'frames_tested': stats['frames_tested']
    }


def main():
    print("🔍 COMPARACIÓN DE MODELOS DE HOMOGRAFÍA")
    print("="*60)

    # Buscar video en inputs
    inputs_dir = Path("inputs")
    videos = list(inputs_dir.glob("*.mp4")) + list(inputs_dir.glob("*.mov")) + list(inputs_dir.glob("*.avi"))

    if not videos:
        print("❌ No se encontraron videos en 'inputs/'")
        return

    video_path = videos[0]
    print(f"📹 Video: {video_path.name}")

    # Modelos a comparar
    models = [
        {
            'path': 'models/homography.pt',
            'name': 'Homography.pt (32 keypoints)'
        },
        {
            'path': 'models/soccana_keypoint/Model/weights/best.pt',
            'name': 'Soccana Keypoint (29 keypoints)'
        }
    ]

    results = []

    for model_info in models:
        model_path = Path(model_info['path'])
        if model_path.exists():
            result = test_model_on_video(video_path, model_path, model_info['name'], num_frames=20)
            results.append(result)
        else:
            print(f"\n❌ Modelo no encontrado: {model_path}")

    # Comparación final
    print(f"\n{'='*60}")
    print("🏆 COMPARACIÓN FINAL")
    print(f"{'='*60}")

    if len(results) >= 2:
        print(f"\n{'Métrica':<40} {'Homography.pt':<20} {'Soccana':<20}")
        print("-"*80)
        print(f"{'Keypoints totales':<40} {results[0]['num_keypoints']:<20} {results[1]['num_keypoints']:<20}")
        print(f"{'Keypoints detectados (conf>0.05)':<40} {results[0]['avg_detected_low']:<20.1f} {results[1]['avg_detected_low']:<20.1f}")
        print(f"{'Keypoints detectados (conf>0.3)':<40} {results[0]['avg_detected_medium']:<20.1f} {results[1]['avg_detected_medium']:<20.1f}")
        print(f"{'Keypoints detectados (conf>0.5)':<40} {results[0]['avg_detected_high']:<20.1f} {results[1]['avg_detected_high']:<20.1f}")
        print(f"{'Tasa éxito homografía':<40} {results[0]['homography_success_rate']:<20.1f}% {results[1]['homography_success_rate']:<20.1f}%")

        # Determinar ganador
        print(f"\n{'='*60}")
        if results[0]['homography_success_rate'] > results[1]['homography_success_rate']:
            winner = "Homography.pt"
            diff = results[0]['homography_success_rate'] - results[1]['homography_success_rate']
        elif results[1]['homography_success_rate'] > results[0]['homography_success_rate']:
            winner = "Soccana"
            diff = results[1]['homography_success_rate'] - results[0]['homography_success_rate']
        else:
            winner = "Empate"
            diff = 0

        if winner != "Empate":
            print(f"🥇 GANADOR: {winner}")
            print(f"   Ventaja: {diff:.1f}% más de homografías exitosas")
        else:
            print(f"🤝 EMPATE: Ambos modelos tienen el mismo rendimiento")

        # Recomendación
        print(f"\n💡 RECOMENDACIÓN:")
        if results[0]['homography_success_rate'] >= 75 and results[0]['homography_success_rate'] > results[1]['homography_success_rate']:
            print(f"   Usa 'homography.pt' en Streamlit")
            print(f"   • Mayor tasa de éxito ({results[0]['homography_success_rate']:.1f}%)")
            print(f"   • {results[0]['num_keypoints']} keypoints (más puntos = más robusto)")
        elif results[1]['homography_success_rate'] >= 75 and results[1]['homography_success_rate'] > results[0]['homography_success_rate']:
            print(f"   Usa 'Soccana Keypoint' en Streamlit")
            print(f"   • Mayor tasa de éxito ({results[1]['homography_success_rate']:.1f}%)")
            print(f"   • Modelo especializado en fútbol")
        else:
            best = results[0] if results[0]['homography_success_rate'] > results[1]['homography_success_rate'] else results[1]
            print(f"   Usa '{best['model_name']}'")
            print(f"   • Es el que mejor funciona en tu video ({best['homography_success_rate']:.1f}%)")
            if best['homography_success_rate'] < 60:
                print(f"   ⚠️  Tasa de éxito baja - considera usar 'Aproximación Pantalla Completa'")


if __name__ == "__main__":
    main()
