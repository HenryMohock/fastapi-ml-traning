from moviepy.video.io.VideoFileClip import VideoFileClip


def trim_video(input_path, output_path, start_time, end_time):
    # Открываем видео файл
    with VideoFileClip(input_path) as video:
        # Обрезаем видео
        trimmed_video = video.subclip(start_time, end_time)
        # Сохраняем обрезанный кусок в новый файл
        trimmed_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


# Пример использования
input_video_path = "../../video/input_video.mp4"
output_video_path = "../../video/output_video.mp4"
start_time = 27  # Время начала обрезки в секундах (2 минута)
end_time = 47   # Время конца обрезки в секундах (3 минута)

trim_video(input_video_path, output_video_path, start_time, end_time)
