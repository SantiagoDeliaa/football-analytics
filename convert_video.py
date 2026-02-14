"""
Script para convertir video de .mkv a .mp4
"""
from moviepy import VideoFileClip

input_file = r'videos\europe_uefa-champions-league\2016-2017\2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG\1_720p.mkv'
output_file = r'videos\europe_uefa-champions-league\2016-2017\2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG\1_720p.mp4'

print(f"Convirtiendo {input_file} a {output_file}...")
clip = VideoFileClip(input_file)
clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
clip.close()
print("¡Conversión completada!")
