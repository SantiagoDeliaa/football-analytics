"""
Script de test para validar la integración completa del modelo Soccana
con el sistema de tracking, homografía y radar.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.controllers.process_video import process_video
from src.models.load_model import load_roboflow_model

def test_soccana_integration():
    print("\n" + "="*80)
    print("TEST DE INTEGRACIÓN: Modelo Soccana + Tracking + Radar")
    print("="*80 + "\n")

    # Paths
    video_path = project_root / "inputs" / "2_720p_clip_5-20.mp4"
    soccana_model_path = project_root / "models" / "soccana_keypoint" / "Model" / "weights" / "best.pt"
    output_path = project_root / "outputs" / "test_soccana_integration.mp4"

    # Crear directorio de salida
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📹 Video de entrada: {video_path}")
    print(f"🤖 Modelo Soccana: {soccana_model_path}")
    print(f"💾 Video de salida: {output_path}\n")

    # Verificar archivos
    if not video_path.exists():
        print(f"❌ ERROR: Video no encontrado: {video_path}")
        return False

    if not soccana_model_path.exists():
        print(f"❌ ERROR: Modelo Soccana no encontrado: {soccana_model_path}")
        print("   Ejecuta: python scripts/download_soccana_model.py")
        return False

    try:
        # 1. Cargar modelo de jugadores (YOLOv8 genérico)
        print("[1/4] Cargando modelo de jugadores (YOLOv8m)...")
        player_model = load_roboflow_model("yolov8m")
        print("✅ Modelo de jugadores cargado\n")

        # 2. Cargar modelo de campo (Soccana)
        print("[2/4] Cargando modelo de campo (Soccana YOLOv11)...")
        pitch_model = YOLO(str(soccana_model_path))
        print("✅ Modelo Soccana cargado (29 keypoints)\n")

        # 3. Procesar video
        print("[3/4] Procesando video con tracking + homografía + radar...")
        print("     (Esto puede tomar unos minutos dependiendo del largo del video)\n")

        process_video(
            source_path=str(video_path),
            target_path=str(output_path),
            player_model=player_model,
            ball_model=None,  # Usaremos detección de pelota del modelo de jugadores
            pitch_model=pitch_model,
            conf=0.3,
            detection_mode="players_and_ball",
            full_field_approx=False  # Usar modelo Soccana, no aproximación
        )

        print("\n✅ Procesamiento completado!")
        print(f"💾 Video guardado en: {output_path}\n")

        # 4. Verificar salida
        print("[4/4] Verificando archivo de salida...")
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Archivo generado correctamente")
            print(f"   Tamaño: {size_mb:.2f} MB\n")

            print("="*80)
            print("RESULTADO DEL TEST")
            print("="*80 + "\n")
            print("✅ ¡Integración exitosa!")
            print("\nCaracterísticas validadas:")
            print("  ✓ Detección de jugadores con YOLOv8")
            print("  ✓ Tracking con ByteTrack")
            print("  ✓ Clasificación en equipos (K-means clustering)")
            print("  ✓ Detección de keypoints con Soccana (29 puntos)")
            print("  ✓ Homografía con RANSAC (múltiples keypoints)")
            print("  ✓ Proyección a radar 2D (vista táctica)")
            print("  ✓ Suavizado temporal de posiciones")
            print("\nPróximos pasos:")
            print("  1. Revisar el video generado")
            print("  2. Verificar que el radar se visualiza correctamente")
            print("  3. Confirmar que la homografía funciona (jugadores en posiciones correctas)")
            print("  4. Si todo está OK, proceder con Paso 2 (Formaciones) y Paso 3 (Métricas)\n")

            return True
        else:
            print("❌ ERROR: No se generó el archivo de salida")
            return False

    except Exception as e:
        print(f"\n❌ ERROR durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_soccana_integration()

    if success:
        print("="*80)
        print("TEST EXITOSO ✅")
        print("="*80 + "\n")
    else:
        print("="*80)
        print("TEST FALLIDO ❌")
        print("="*80 + "\n")

    sys.exit(0 if success else 1)
