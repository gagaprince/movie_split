from movie_util import parse_clip, save_clip, get_movie_time, movie_split_by_time, movie_scale_speed, add_movie, movie_curve_change_scale, movie_curve_chang_speed, movie_old_film
from audio_util import get_beat_times, parse_audio_clip


# if __name__ == '__main__':
#     times = get_beat_times("./movie/2.mp3")
#     print(times)

if __name__ == '__main__':
    in_file = "./movie/1.mp4"
    mp3_file = "./movie/2.mp3"
    out_file = "./movie/out.mp4"
    print(f"对{in_file}进行处理，从2s开始进行曲线变速")
    video_clip = parse_clip(in_file)
    audio_clip = parse_audio_clip(mp3_file)
    movie_time = get_movie_time(video_clip)
    print(f"视频时长: {movie_time} 秒")

    full_clip = movie_curve_change_scale(video_clip, 1.64861678, 6.80344671, 0.464, 1.5)

    # full_clip = movie_old_film(full_clip)

    full_clip = movie_curve_chang_speed(full_clip, 1.64861678, 6.80344671, 0.464, 2, 0.66)

    full_clip = movie_split_by_time(full_clip,0, 7)

    full_clip = full_clip.set_audio(audio_clip)

    save_clip(full_clip, out_file)

# if __name__ == '__main__':
#     in_file = "./movie/1.mp4"
#     start_time = 3
#     end_time = 5
#     out_split_file = "./movie/split_3_5.mp4"
#     out_speed_2_file = "./movie/speed_2.mp4"
#     print(f"视频路径: {in_file}")
#     video_clip = parse_clip(in_file)
#     movie_time = get_movie_time(video_clip)
#     print(f"视频时长: {movie_time} 秒")
#
#     print(f"现将视频进行截取，截取{start_time}s到{end_time}s,保存在{out_split_file}")
#     split_clip = movie_split_by_time(video_clip, start_time, end_time)
#     save_clip(split_clip, out_split_file)
#
#     print(f"现将视频进行变速，2倍速，保存在{out_speed_2_file}")
#     speed_clip = movie_scale_speed(split_clip, 2)
#     save_clip(speed_clip, out_speed_2_file)

# if __name__ == '__main__':
#     in_file = "./movie/split_3_5.mp4"
#     out_file = "./movie/out.mp4"
#     print(f"对{in_file}进行处理，前一秒2倍速后一秒0.5倍速")
#     video_clip = parse_clip(in_file)
#     movie_time = get_movie_time(video_clip)
#     print(f"视频时长: {movie_time} 秒")
#
#     split_1_clip = movie_split_by_time(video_clip,0,1)
#     split_2_clip = movie_split_by_time(video_clip, 1, 2)
#
#     speed_split_1_clip = movie_scale_speed(split_1_clip, 2)
#     speed_split_2_clip = movie_scale_speed(split_2_clip, 0.66)
#
#     full_clip = add_movie([speed_split_1_clip, speed_split_2_clip])
#
#     save_clip(full_clip, out_file)







