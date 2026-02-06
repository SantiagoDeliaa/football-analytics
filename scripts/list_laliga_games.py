"""
Script para listar partidos disponibles de La Liga en SoccerNet
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader

# Cargar variables de entorno
load_dotenv()

def list_laliga_games(output_dir="data/soccernet_metadata"):
    """Lista todos los partidos disponibles de La Liga"""

    # Crear directorio temporal
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(out_path))

    # Configurar password
    password = os.getenv("SOCCERNET_PASSWORD")
    if password:
        downloader.password = password

    print("Descargando metadata de La Liga...")
    print("=" * 60)

    # Descargar solo los labels para ver qué partidos hay
    try:
        # Primero intentamos descargar labels de todos los splits
        for split in ["train", "valid", "test"]:
            print(f"\nSplit: {split}")
            try:
                downloader.downloadGames(
                    files=["Labels-v2.json"],
                    split=[split]
                )

                # Listar partidos de La Liga encontrados
                laliga_games = []
                for match_dir in out_path.rglob("spain_laliga/*"):
                    if match_dir.is_dir() and (match_dir / "Labels-v2.json").exists():
                        relative_path = match_dir.relative_to(out_path)
                        laliga_games.append(str(relative_path))

                if laliga_games:
                    print(f"  Partidos encontrados: {len(laliga_games)}")
                    for game in sorted(laliga_games)[:10]:  # Mostrar primeros 10
                        print(f"    - {game}")
                    if len(laliga_games) > 10:
                        print(f"    ... y {len(laliga_games) - 10} más")
                else:
                    print(f"  No se encontraron partidos de La Liga en {split}")

            except Exception as e:
                print(f"  Error en split {split}: {e}")

    except Exception as e:
        print(f"Error general: {e}")

    print("\n" + "=" * 60)
    print(f"Metadata guardada en: {out_path.absolute()}")

if __name__ == "__main__":
    list_laliga_games()
