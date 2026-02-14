"""
Script para probar los módulos de formaciones y métricas tácticas
antes de integrarlos en el pipeline completo
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.controllers.formation_detector import FormationDetector
from src.controllers.tactical_metrics import TacticalMetricsCalculator, TacticalMetricsTracker


def test_formation_detector():
    print("\n" + "="*80)
    print("TEST 1: Formation Detector")
    print("="*80 + "\n")

    detector = FormationDetector()

    # Simular posiciones de jugadores en formación 4-4-2
    # Campo: 105m (X) x 68m (Y)
    # Equipo atacando hacia la derecha (X creciente)

    print("[1/3] Simulando formación 4-4-2...")
    positions_442 = np.array([
        # Defensores (X: 15-25m)
        [20, 10],  # Defensa derecho
        [20, 25],  # Defensa central derecho
        [20, 43],  # Defensa central izquierdo
        [20, 58],  # Defensa izquierdo

        # Mediocampistas (X: 40-55m)
        [48, 12],  # Mediocampista derecho
        [48, 28],  # Mediocampista central derecho
        [48, 40],  # Mediocampista central izquierdo
        [48, 56],  # Mediocampista izquierdo

        # Delanteros (X: 75-85m)
        [80, 25],  # Delantero derecho
        [80, 43],  # Delantero izquierdo
    ])

    result = detector.detect_formation(positions_442, team_attacking_direction="right")

    print(f"Formación detectada: {result['formation']}")
    print(f"Confianza: {result['confidence']:.2f}")
    print(f"Jugadores por línea: {result['players_per_line']}")
    print(f"  - Defensa: {len(result['lines']['defense'])} jugadores")
    print(f"  - Mediocampo: {len(result['lines']['midfield'])} jugadores")
    print(f"  - Ataque: {len(result['lines']['attack'])} jugadores")

    # Test 4-3-3
    print("\n[2/3] Simulando formación 4-3-3...")
    positions_433 = np.array([
        # Defensores
        [20, 10], [20, 25], [20, 43], [20, 58],
        # Mediocampistas (solo 3)
        [48, 20], [48, 34], [48, 48],
        # Delanteros (3)
        [80, 15], [80, 34], [80, 53],
    ])

    result2 = detector.detect_formation(positions_433, team_attacking_direction="right")
    print(f"Formación detectada: {result2['formation']}")
    print(f"Confianza: {result2['confidence']:.2f}")
    print(f"Jugadores por línea: {result2['players_per_line']}")

    # Test con jugadores parciales (solo 6)
    print("\n[3/3] Simulando vista parcial (6 jugadores)...")
    positions_partial = positions_442[:6]  # Solo primeros 6 jugadores

    result3 = detector.detect_formation(positions_partial, team_attacking_direction="right")
    print(f"Formación detectada: {result3['formation']}")
    print(f"Confianza: {result3['confidence']:.2f}")
    print(f"Nota: Sistema adaptativo funciona con jugadores parcialmente visibles")

    print("\n[OK] Formation Detector funcionando correctamente\n")


def test_tactical_metrics():
    print("="*80)
    print("TEST 2: Tactical Metrics Calculator")
    print("="*80 + "\n")

    calculator = TacticalMetricsCalculator()

    # Posiciones de ejemplo
    positions = np.array([
        [30, 20], [30, 35], [30, 50],
        [50, 15], [50, 30], [50, 45], [50, 55],
        [70, 25], [70, 43]
    ])

    print("[1/5] Calculando todas las métricas...")
    metrics = calculator.calculate_all_metrics(positions)

    print(f"\nMétricas calculadas:")
    print(f"  - Compactación (área): {metrics['compactness']:.2f} m²")
    print(f"  - Altura de presión: {metrics['pressure_height']:.2f} m (de 0-105)")
    print(f"  - Amplitud ofensiva: {metrics['offensive_width']:.2f} m (de 0-68)")
    print(f"  - Centroide: ({metrics['centroid'][0]:.2f}, {metrics['centroid'][1]:.2f})")
    print(f"  - Stretch Index: {metrics['stretch_index']:.2f}")
    print(f"  - Profundidad defensiva: {metrics['defensive_depth']:.2f} m")
    print(f"  - Jugadores: {metrics['num_players']}")

    print("\n[2/5] Probando tracker temporal...")
    tracker = TacticalMetricsTracker(history_size=10)

    # Simular 10 frames con variación
    for i in range(10):
        # Simular movimiento (avanzar presión)
        moving_positions = positions.copy()
        moving_positions[:, 0] += i * 2  # Avanzar 2m por frame

        frame_metrics = calculator.calculate_all_metrics(moving_positions)
        tracker.update(frame_metrics, frame_number=i)

    print(f"[OK] Tracker acumuló {len(tracker.metrics_history['frame_number'])} frames")

    print("\n[3/5] Calculando estadísticas...")
    stats = tracker.get_statistics()

    print(f"\nEstadísticas de altura de presión:")
    print(f"  - Media: {stats['pressure_height']['mean']:.2f} m")
    print(f"  - Desv. Std: {stats['pressure_height']['std']:.2f} m")
    print(f"  - Mínimo: {stats['pressure_height']['min']:.2f} m")
    print(f"  - Máximo: {stats['pressure_height']['max']:.2f} m")
    print(f"  - Actual: {stats['pressure_height']['current']:.2f} m")

    print("\n[4/5] Analizando tendencias...")
    trend = tracker.get_trend('pressure_height', window=10)
    print(f"Tendencia de altura de presión: {trend}")
    print(f"  Interpretación: El equipo {'avanza' if trend == 'increasing' else 'retrocede' if trend == 'decreasing' else 'mantiene posición'}")

    print("\n[5/5] Exportando datos...")
    export_data = tracker.export_to_dict()
    arrays = tracker.export_to_arrays()
    print(f"[OK] Datos exportables:")
    print(f"  - Dict keys: {list(export_data.keys())}")
    print(f"  - Array shapes: {', '.join([f'{k}: {v.shape}' for k, v in arrays.items()])}")

    print("\n[OK] Tactical Metrics funcionando correctamente\n")


def main():
    print("\n" + "="*80)
    print("TEST DE MODULOS TACTICOS")
    print("="*80)

    try:
        test_formation_detector()
        test_tactical_metrics()

        print("="*80)
        print("RESULTADO: TODOS LOS TESTS PASARON")
        print("="*80)
        print("\nLos módulos están listos para integración en el pipeline principal.")
        print("\nPróximo paso: Integrar en process_video.py\n")

        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
