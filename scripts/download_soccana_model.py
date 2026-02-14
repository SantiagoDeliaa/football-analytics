"""
Script para descargar el modelo Soccana_Keypoint de HuggingFace
YOLOv11 con 29 keypoints de campo de futbol
"""

import sys
from pathlib import Path

def download_soccana_model():
    print("\n" + "="*70)
    print("DESCARGA DE MODELO: Soccana_Keypoint (YOLOv11)")
    print("="*70 + "\n")

    print("Modelo: Adit-jain/Soccana_Keypoint")
    print("Arquitectura: YOLOv11 Pose")
    print("Keypoints: 29 puntos del campo de futbol")
    print("Uso: Deteccion de lineas y esquinas del campo\n")

    try:
        # Intentar con huggingface_hub
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            print("[OK] huggingface_hub disponible")
        except ImportError:
            print("[INFO] Instalando huggingface_hub...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            from huggingface_hub import hf_hub_download, snapshot_download
            print("[OK] huggingface_hub instalado")

        output_dir = Path("models/soccana_keypoint")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n[1/2] Descargando repositorio completo...")
        print("      (Esto incluira el modelo .pt y archivos de configuracion)\n")

        # Descargar todo el repositorio
        repo_path = snapshot_download(
            repo_id="Adit-jain/Soccana_Keypoint",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False
        )

        print(f"\n[OK] Repositorio descargado en: {output_dir}\n")

        # Buscar archivos .pt
        print("[2/2] Buscando archivos de modelo (.pt)...\n")
        pt_files = list(output_dir.rglob("*.pt"))

        if pt_files:
            print(f"[OK] Encontrados {len(pt_files)} archivos .pt:\n")
            for pt_file in pt_files:
                rel_path = pt_file.relative_to(output_dir)
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                print(f"  - {rel_path} ({size_mb:.1f} MB)")

            # Buscar el mejor modelo (best.pt)
            best_models = [f for f in pt_files if 'best' in f.name.lower()]

            if best_models:
                best_model = best_models[0]
                print(f"\n[RECOMENDADO] Usar este modelo:")
                print(f"  {best_model.relative_to(output_dir)}\n")
                return best_model
            else:
                print(f"\n[INFO] Usar el primer modelo encontrado:")
                print(f"  {pt_files[0].relative_to(output_dir)}\n")
                return pt_files[0]
        else:
            print("[ADVERTENCIA] No se encontraron archivos .pt")
            print("\nArchivos descargados:")
            for item in output_dir.rglob("*"):
                if item.is_file():
                    print(f"  - {item.relative_to(output_dir)}")

            print("\n[INFO] El repositorio puede contener codigo para entrenar el modelo")
            print("       o instrucciones para descargarlo de otra fuente.\n")
            return output_dir

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_path = download_soccana_model()

    if model_path:
        print("\n" + "="*70)
        print("DESCARGA COMPLETADA")
        print("="*70 + "\n")

        if model_path.suffix == '.pt':
            print(f"Modelo guardado en: {model_path}\n")
            print("Proximo paso:")
            print("  1. Probar el modelo con tu video:")
            print(f"     python scripts/validate_pitch_model.py \\")
            print(f"       --model \"{model_path}\" \\")
            print(f"       --video \"inputs/2_720p_clip_5-20.mp4\"\n")
            print("  2. Si funciona bien, integrar en app.py")
        else:
            print(f"Repositorio descargado en: {model_path}\n")
            print("Revisar la documentacion del repositorio para:")
            print("  1. Instrucciones de entrenamiento")
            print("  2. Enlaces a modelos pre-entrenados")
            print("  3. Ejemplos de uso")
    else:
        print("\n[ERROR] Descarga fallida")
        sys.exit(1)
