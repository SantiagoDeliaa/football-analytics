"""
Script de prueba para verificar qué videos están disponibles en SoccerNet
"""

import os
from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader
from pathlib import Path

load_dotenv()

# Configurar downloader
downloader = SoccerNetDownloader(LocalDirectory="data/soccernet_test")
password = os.getenv("SOCCERNET_PASSWORD")
if password:
    downloader.password = password

# Intentar descargar un video conocido de la Premier League (Inglaterra)
print("Intentando descargar un video de prueba de la Premier League...")
try:
    downloader.downloadGame(
        files=["1_224p.mkv"],
        game="england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
    )
    print("Descarga exitosa!")

    # Verificar si el archivo existe
    video_path = Path("data/soccernet_test/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv")
    if video_path.exists():
        print(f"Video encontrado en: {video_path}")
        print(f"Tamaño: {video_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print("El video no se descargo correctamente")

except Exception as e:
    print(f"Error: {e}")
    print("\nNota: Es posible que los videos no esten disponibles con tu credencial,")
    print("o que necesites aceptar un NDA (Non-Disclosure Agreement) en SoccerNet")
