"""
Verifica la estructura del archivo stats.json y lo muestra en formato legible
"""
import json
from pathlib import Path

stats_files = list(Path("outputs").glob("*_stats.json"))

if not stats_files:
    print("❌ No se encontraron archivos *_stats.json en outputs/")
    exit(1)

print(f"Encontrados {len(stats_files)} archivos de estadisticas\n")

for stats_file in stats_files:
    print("="*70)
    print(f"Archivo: {stats_file.name}")
    print("="*70)

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Verificar estructura
    required_keys = ['total_frames', 'duration_seconds', 'formations', 'metrics', 'timeline']

    print("\nVerificacion de estructura:")
    for key in required_keys:
        if key in stats:
            print(f"  [OK] {key}: Presente")
        else:
            print(f"  [FALTA] {key}: FALTA")

    # Mostrar resumen
    print(f"\nResumen:")
    print(f"  Total frames: {stats.get('total_frames', 'N/A')}")
    print(f"  Duración: {stats.get('duration_seconds', 0):.1f}s")

    if 'formations' in stats:
        print(f"\n  Formaciones:")
        form1 = stats['formations'].get('team1', {}).get('most_common', 'N/A')
        form2 = stats['formations'].get('team2', {}).get('most_common', 'N/A')
        print(f"    Team 1: {form1}")
        print(f"    Team 2: {form2}")

    if 'metrics' in stats:
        print(f"\n  Métricas promedio disponibles:")
        metrics1 = stats['metrics'].get('team1', {})
        if metrics1:
            for metric_name in metrics1.keys():
                print(f"    • {metric_name}")
        else:
            print(f"    [ERROR] No hay metricas para Team 1")

    if 'timeline' in stats:
        timeline1 = stats['timeline'].get('team1', {})
        if timeline1 and 'frame_number' in timeline1:
            print(f"\n  Timeline: {len(timeline1['frame_number'])} frames registrados")
        else:
            print(f"\n  [ERROR] Timeline vacio o sin estructura correcta")

    print()

print("="*70)
print("\nNota:")
print("  Si alguna clave FALTA, el archivo tiene estructura antigua.")
print("  Solucion: Elimina el archivo y procesa el video nuevamente.")
print("="*70)
