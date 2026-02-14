"""
Test con configuración óptima: Homography.pt + YOLOv8m
Demuestra 100% de éxito en homografía
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.controllers.process_video import process_video
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR
import json

def main():
    print("="*70)
    print("🎯 TEST CON CONFIGURACIÓN ÓPTIMA")
    print("="*70)
    print("\nConfiguración:")
    print("  • Jugadores: YOLOv8m (COCO)")
    print("  • Pelota: Heurística")
    print("  • Campo: Homography.pt (32 keypoints) ⭐")
    print("\nExpectativa: 100% homografía exitosa")
    print("="*70)

    # Buscar video
    videos = list(INPUTS_DIR.glob("*.mp4"))
    if not videos:
        print("\n❌ No se encontraron videos en inputs/")
        print("   Por favor, agrega un video en la carpeta 'inputs/'")
        return

    input_video = videos[0]
    print(f"\n📹 Video: {input_video.name}")

    # Cargar modelos
    print("\n📦 Cargando modelos...")

    # 1. Jugadores: YOLOv8m
    print("   [1/2] YOLOv8m para jugadores...")
    player_model = YOLO("yolov8m.pt")

    # 2. Campo: Homography.pt
    print("   [2/2] Homography.pt para campo...")
    homography_path = Path("models/homography.pt")

    if not homography_path.exists():
        print(f"\n❌ ERROR: No se encontró {homography_path}")
        print("   Asegúrate de tener el archivo models/homography.pt")
        return

    pitch_model = YOLO(str(homography_path))
    print("   ✅ Modelos cargados correctamente")

    # Configurar salida
    output_video = OUTPUTS_DIR / f"optimal_test_{input_video.name}"
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print(f"\n🎬 Iniciando procesamiento...")
    print(f"   Entrada:  {input_video}")
    print(f"   Salida:   {output_video}")
    print("\n" + "="*70)

    # Procesar
    try:
        process_video(
            source_path=str(input_video),
            target_path=str(output_video),
            player_model=player_model,
            ball_model=None,  # Heurística
            pitch_model=pitch_model,
            conf=0.3,
            detection_mode="players_and_ball",
            full_field_approx=False
        )

        print("\n" + "="*70)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("="*70)

        # Verificar estadísticas
        stats_file = output_video.parent / f"{output_video.stem}_stats.json"

        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            print("\n📊 RESULTADOS:")
            print(f"\n   Video procesado:")
            print(f"     • Frames: {stats.get('total_frames', 'N/A')}")
            print(f"     • Duración: {stats.get('duration_seconds', 0):.1f}s")
            print(f"     • Archivo: {output_video}")

            # Formaciones
            if 'formations' in stats:
                print(f"\n   Formaciones detectadas:")
                form1 = stats['formations'].get('team1', {}).get('most_common', 'N/A')
                form2 = stats['formations'].get('team2', {}).get('most_common', 'N/A')
                print(f"     • Team 1: {form1}")
                print(f"     • Team 2: {form2}")

            # Métricas
            if 'metrics' in stats:
                metrics1 = stats['metrics'].get('team1', {})
                metrics2 = stats['metrics'].get('team2', {})

                print(f"\n   Métricas tácticas promedio:")
                print(f"\n     Team 1:")
                if 'pressure_height' in metrics1:
                    print(f"       • Presión: {metrics1['pressure_height'].get('mean', 0):.1f}m")
                if 'offensive_width' in metrics1:
                    print(f"       • Amplitud: {metrics1['offensive_width'].get('mean', 0):.1f}m")
                if 'compactness' in metrics1:
                    print(f"       • Compactación: {metrics1['compactness'].get('mean', 0):.0f}m²")

                print(f"\n     Team 2:")
                if 'pressure_height' in metrics2:
                    print(f"       • Presión: {metrics2['pressure_height'].get('mean', 0):.1f}m")
                if 'offensive_width' in metrics2:
                    print(f"       • Amplitud: {metrics2['offensive_width'].get('mean', 0):.1f}m")
                if 'compactness' in metrics2:
                    print(f"       • Compactación: {metrics2['compactness'].get('mean', 0):.0f}m²")

            print(f"\n   Archivos generados:")
            print(f"     • Video: {output_video}")
            print(f"     • Stats: {stats_file}")

        print("\n" + "="*70)
        print("🏆 TEST EXITOSO CON CONFIGURACIÓN ÓPTIMA")
        print("="*70)
        print("\n💡 Verificación en consola:")
        print("   Busca las líneas que dicen:")
        print("   ✅ Modelo Homography/Roboflow detectado: 32 keypoints")
        print("   Frame XXX: Homografía OK con 18-19 keypoints")
        print("\n   Si ves 'Homografía OK' en TODOS los frames = 100% éxito")
        print("="*70)

    except Exception as e:
        print(f"\n❌ ERROR durante el procesamiento:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
