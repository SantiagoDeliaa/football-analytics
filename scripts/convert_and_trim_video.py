"""
Script para descargar ffmpeg, recortar video y convertir a MP4
"""

import os
import subprocess
import zipfile
from pathlib import Path
import urllib.request
import shutil

def download_ffmpeg():
    """Descarga ffmpeg para Windows si no existe"""
    ffmpeg_dir = Path("tools/ffmpeg")
    ffmpeg_exe = ffmpeg_dir / "bin" / "ffmpeg.exe"

    if ffmpeg_exe.exists():
        print(f"ffmpeg ya existe en: {ffmpeg_exe}")
        return str(ffmpeg_exe)

    print("Descargando ffmpeg...")
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    # URL de ffmpeg essentials build
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = ffmpeg_dir / "ffmpeg.zip"

    print(f"Descargando desde {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extrayendo ffmpeg...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ffmpeg_dir)

    # Buscar el ejecutable en la estructura extraída
    for item in ffmpeg_dir.rglob("ffmpeg.exe"):
        # Copiar a la ubicación esperada
        bin_dir = ffmpeg_dir / "bin"
        bin_dir.mkdir(exist_ok=True)
        shutil.copy(item, bin_dir / "ffmpeg.exe")
        print(f"ffmpeg instalado en: {bin_dir / 'ffmpeg.exe'}")

        # Limpiar archivo zip
        zip_path.unlink()
        return str(bin_dir / "ffmpeg.exe")

    raise Exception("No se pudo encontrar ffmpeg.exe en el archivo descargado")


def trim_and_convert_video(input_path, output_path, start_time="00:00:00", end_time="00:00:41"):
    """
    Recorta y convierte video de MKV a MP4

    Args:
        input_path: Ruta del video de entrada (MKV)
        output_path: Ruta del video de salida (MP4)
        start_time: Tiempo de inicio (formato HH:MM:SS)
        end_time: Tiempo de fin (formato HH:MM:SS)
    """

    # Descargar/obtener ffmpeg
    ffmpeg_path = download_ffmpeg()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Video de entrada no encontrado: {input_path}")

    # Crear directorio de salida si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRecortando video:")
    print(f"  Entrada: {input_path}")
    print(f"  Salida: {output_path}")
    print(f"  Desde: {start_time}")
    print(f"  Hasta: {end_time}")

    # Comando ffmpeg para recortar y convertir
    # -ss: tiempo de inicio
    # -to: tiempo de fin
    # -i: archivo de entrada
    # -c:v copy: copiar video sin recodificar (más rápido)
    # -c:a copy: copiar audio sin recodificar
    cmd = [
        ffmpeg_path,
        "-ss", start_time,
        "-to", end_time,
        "-i", str(input_path),
        "-c:v", "libx264",  # Recodificar a H.264 para MP4
        "-c:a", "aac",      # Recodificar audio a AAC
        "-y",               # Sobrescribir si existe
        str(output_path)
    ]

    print(f"\nEjecutando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\nVideo convertido exitosamente!")
        print(f"Tamaño: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"Ubicación: {output_path.absolute()}")
    else:
        print(f"Error al convertir video:")
        print(result.stderr)
        raise Exception("Error en conversión de video")


if __name__ == "__main__":
    # Video de entrada (720p)
    input_video = "videos/europe_uefa-champions-league/2016-2017/2016-10-19 - 21-45 Barcelona 4 - 0 Manchester City/1_720p.mkv"

    # Video de salida
    output_video = "videos/barcelona_vs_mancity_00-41.mp4"

    trim_and_convert_video(
        input_path=input_video,
        output_path=output_video,
        start_time="00:00:00",
        end_time="00:00:41"
    )
