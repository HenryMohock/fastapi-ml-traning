import tkinter as tk
from tkinter import filedialog
from moviepy.video.io.VideoFileClip import VideoFileClip


def trim_video(input_path, output_path, start_time, end_time):
    # Открываем видео файл
    with VideoFileClip(input_path) as video:
        # Обрезаем видео
        trimmed_video = video.subclip(start_time, end_time)
        # Сохраняем обрезанный кусок в новый файл
        trimmed_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


def main():
    # Создаем окно Tkinter и скрываем его
    root = tk.Tk()
    root.withdraw()

    # Открываем диалог для выбора входного видео файла
    input_video_path = filedialog.askopenfilename(title="Выберите видео файл для обрезки",
                                                  filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])

    if not input_video_path:
        print("Не выбран входной видео файл.")
        return

    # Открываем диалог для выбора места сохранения обрезанного видео файла
    output_video_path = filedialog.asksaveasfilename(title="Сохранить обрезанное видео как",
                                                     defaultextension=".mp4",
                                                     filetypes=[("MP4 files", "*.mp4"),
                                                                ("AVI files", "*.avi"),
                                                                ("MOV files", "*.mov"),
                                                                ("MKV files", "*.mkv")])

    if not output_video_path:
        print("Не выбрано место сохранения обрезанного видео файла.")
        return

    # Запрашиваем у пользователя начальное и конечное время обрезки
    start_time = float(input("Введите начальное время обрезки (в секундах): "))
    end_time = float(input("Введите конечное время обрезки (в секундах): "))

    # Выполняем обрезку видео
    trim_video(input_video_path, output_video_path, start_time, end_time)

    print(f"Видео обрезано и сохранено как {output_video_path}")


if __name__ == "__main__":
    main()
