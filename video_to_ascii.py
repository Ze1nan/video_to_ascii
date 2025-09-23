from pydantic import BaseModel, field_validator, model_validator, FilePath, computed_field
from typing import Optional, Tuple
from moviepy import VideoFileClip
from playsound import playsound
from tqdm import tqdm
from PIL import Image
import threading
import argparse
import shutil
import pickle
import time
import sys
import cv2
import os

class Config(BaseModel):
    video: Optional[FilePath] = None
    backup: Optional[str] = None
    audio: Optional[FilePath] = None
    symbols: Optional[str] = None
    fps: Optional[int] = None
    resolution: Optional[int] = None
    max_width: Optional[int] = None 
    max_height: Optional[int] = None
    high_quality: Optional[bool] = None

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        if v:
            symbols = {}
            try:
                pairs = v.split(',')
                for pair in pairs:
                    symbol, threshold = pair.split(':')
                    symbols[symbol] = int(threshold)
            except ValueError:
                print("Ошибка: неверный формат для символов. Используйте: символ:порог, например: @:200,+:100,-:75")
                sys.exit(1)
            return dict(sorted(symbols.items(), key=lambda item: item[1], reverse=True))
        return {
            '@': 200,
            '+': 100,
            '-': 75,
            ',': 50,
            '.': 20,
        }
    @field_validator('resolution')
    @classmethod
    def validate_resolution(cls, v):
        if v:
            return list(map(int, v.split(':')))
        return None
    
    @computed_field
    def terminal_window_size(self) -> Tuple[int]:
        terminal_size = shutil.get_terminal_size()
        return (terminal_size.columns, terminal_size.lines)

    @model_validator(mode='after')
    def validate_model(self):
        if self.video is None and self.backup is None:
            raise ValueError("Ошибка: необходимо указать либо видеофайл, либо файл бэкапа.")
        
        if self.max_width and self.max_height:
            raise ValueError("Ошибка: можно указать только одно из двух: либо max_width, либо max_height")
        elif self.max_width:
            if self.max_width > self.terminal_window_size[0]:
                print("Предупреждение: max_width больше ширины вашего терминала")
        elif self.max_height:
            if self.max_height > self.terminal_window_size[1]:
                print("Предупреждение: max_height больше высоты вашего терминала")
        else:
            self.max_width = 120

        if self.audio is None and self.video is not None:
            video_clip = VideoFileClip(self.video)
            if video_clip.audio:
                video_clip.audio.write_audiofile('temp.mp3', logger=None)
                self.audio = 'temp.mp3'

        return self


def load_backup(backup):
    if backup and os.path.exists(backup):
        with open(backup, 'rb') as f:
            raw_frames = pickle.load(f)
            frames = [[list(s) for s in frame.split('\n')] for frame in raw_frames]
        return frames

def create_frames(
    config: Config
):
    cap = cv2.VideoCapture(config.video)

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    aspect_ratio = video_width / video_height

    if not config.resolution:
        if config.max_width:
            while video_width % config.max_width > config.max_width//2:
                config.max_width -= 1
            symbols_width = config.max_width
            symbols_height = int(symbols_width // aspect_ratio) // 2
        elif config.max_height:
            while video_height % config.max_height > config.max_height//2:
                config.max_height -= 1
            symbols_height = config.max_height
            symbols_width = round(symbols_height * aspect_ratio) * 2
        else:
            if video_width > video_height:
                symbols_height = config.terminal_window_size[0]
    else:
        symbols_width, symbols_height = config.resolution


    sector_width, sector_height = (video_width//symbols_width, video_height//symbols_height)
    
    frames = []

    for _ in tqdm(range(total_frames), desc='Load frames'):
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_list = [[' ' for _ in range(symbols_width)] for _ in range(symbols_height)]
        for sector_y in range(0, video_height, sector_height):
            for sector_x in range(0, video_width, sector_width):
                avg_r, avg_g, avg_b = (None, None, None)

                if not config.high_quality:
                    center_x = sector_x + sector_width // 2
                    center_y = sector_y + sector_height // 2

                    # Получаем цвет центрального пикселя
                    try:
                        central_pixel = pil_image.getpixel((center_x, center_y))
                    except IndexError:
                        continue
                    avg_r, avg_g, avg_b = central_pixel
                else:
                    box = (sector_x, sector_y, sector_x + sector_width, sector_y + sector_height)

                    sector = pil_image.crop(box) 

                    total_r, total_g, total_b = 0, 0, 0
                    pixel_count = 0
                    
                    for pixel in sector.getdata():
                        r, g, b = pixel
                        total_r += r
                        total_g += g
                        total_b += b
                        pixel_count += 1

                    if pixel_count > 0:
                        avg_r = total_r // pixel_count
                        avg_g = total_g // pixel_count
                        avg_b = total_b // pixel_count

                try:
                    for symbol, threshold in config.symbols.items():
                        if avg_r >= threshold and avg_g >= threshold and avg_b >= threshold:
                            frame_list[sector_y // sector_height][sector_x // sector_width] = symbol
                            break
                except Exception:
                    pass
        frames.append(frame_list)

    video_clip = VideoFileClip(config.video)
    config.fps = len(frames) / video_clip.duration
    
    cap.release()
    if config.backup:
        with open(config.backup, 'wb') as f:
            data = {
                'config': config,
                'frames': ["\n".join("".join(row) for row in u) for u in frames]
            }
            pickle.dump(data, f)
    return frames

def main():
    parser = argparse.ArgumentParser("Video to ASCII-art")
    parser.add_argument('-v', '--video', type=str, help='Путь к видеофайлу для конвертации', required=False)
    parser.add_argument('-b', '--backup', type=str, help='Путь к файлу бэкапа с кадрами', required=False)
    parser.add_argument('-a', '--audio', type=str, help='Путь к аудиофайлу', required=False)
    parser.add_argument('-s', '--symbols', type=str, help='Символы и пороги в формате: символ:порог, например: @:200,+:100,-:75', required=False)
    parser.add_argument('-fps', '--fps', type=str, help='Количество кадров в секунду в видео', required=False)
    parser.add_argument('-r', '--resolution', type=str, help='Разрешение экрана в формате width:height', required=False)
    parser.add_argument('--max-width', type=int, help='Максимальная ширина видео в символах', required=False)
    parser.add_argument('--max-height', type=int, help='Максимальная высота видео в символах', required=False)
    parser.add_argument('--high-quality', action='store_true', help='Используется более затратный и долгий, но более точный алгоритм')

    args = parser.parse_args()

    try:
        config = Config(
            video=args.video,
            backup=args.backup,
            audio=args.audio,
            symbols=args.symbols,
            fps=args.fps,
            resolution=args.resolution,
            max_width=args.max_width,
            max_height=args.max_height,
            high_quality=args.high_quality
        )
    except ValueError as e:
        print(e)
        sys.exit(1)

    
    if os.path.exists(config.backup):
        frames = load_backup(config.backup)
    else:
        frames = create_frames(config)
    
    input('Press any key to start video')

    if config.audio:
        music_thread = threading.Thread(target=playsound, args=(os.path.abspath(config.audio),))
        music_thread.start()
        time.sleep(1)

    config.fps = 60
    
    os.system('cls')
    for frame in frames:
        s = time.perf_counter()
        for i, row in enumerate(frame):
            sys.stdout.write(f'\033[{i};0H' + "".join(row))
        sys.stdout.flush()
        e = time.perf_counter()-s
        if e < 1/config.fps:
            time.sleep(1/config.fps - e)
    os.system('cls')

if __name__ == "__main__":
    main()
