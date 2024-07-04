from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import cv2


def parse_clip(in_file):
    return VideoFileClip(in_file)


def save_clip(clip, out_file):
    clip.write_videofile(out_file, codec='libx264',  audio_codec='aac')


def get_movie_time(clip):
    return clip.duration


def movie_split_by_time(video_clip, start, end):
    trimmed_clip = video_clip.subclip(start, end)
    return trimmed_clip


def movie_scale_speed(video_clip, scale):
    return video_clip.speedx(scale)


def add_movie(clips):
    return concatenate_videoclips(clips)


# 要求 scale_max>1 scale_min<1
# 需要计算分割点
# 假设分割点比例是x 则 x/scale_max+(1-x)/scale_min = 1 求得x为 (1-1/scale_min)/(1/scale_max - 1/scale_min)
# 0.5/2 + 0.5/scale_min = 1 scale_min = 0.5/0.75 = 0.66
def movie_curve_change_speed_work_step(video_clip, scale_max, scale_min):
    move_time = get_movie_time(video_clip)

    split_time = move_time * (1 - 1 / scale_min) / (1 / scale_max - 1 / scale_min)

    split_1_clip = movie_split_by_time(video_clip, 0, split_time)
    split_2_clip = movie_split_by_time(video_clip, split_time, move_time)

    speed_split_1_clip = movie_scale_speed(split_1_clip, scale_max)
    speed_split_2_clip = movie_scale_speed(split_2_clip, scale_min)

    full_clip = add_movie([speed_split_1_clip, speed_split_2_clip])
    return full_clip


def movie_curve_change_speed_work(video_clip, step, scale_max, scale_min):
    move_time = get_movie_time(video_clip)
    ret_clips = []
    start_time = 0
    while start_time < move_time:
        end_time = start_time + step
        # 确保结束时间不超过视频总时长
        if end_time > move_time:
            end_time = move_time
        subclip = movie_split_by_time(video_clip, start_time, end_time)
        subclip = movie_curve_change_speed_work_step(subclip, scale_max, scale_min)
        ret_clips.append(subclip)
        # 保存分割后的视频片段
        # 更新起始时间
        start_time += step

    return add_movie(ret_clips)


# 曲线变速动画
# video_clip 要剪辑的视频片段
# start 开始时间
# end 结束时间 一般情况下 end-start 是整数个 step
# step 单步距离
# scale_max 高变速
# scale_min 低变速
# return 返回合成的clip
def movie_curve_chang_speed(video_clip, start, end, step, scale_max, scale_min):
    movie_time = get_movie_time(video_clip)
    # 开始和结束clip 不动 最后进行拼接 中间作业的clip 传入另一个方法中进行剪辑
    split_begin_clip = movie_split_by_time(video_clip, 0, start)
    split_end_clip = movie_split_by_time(video_clip, end, movie_time)
    split_work_clip = movie_split_by_time(video_clip, start, end)

    split_work_clip_ret = movie_curve_change_speed_work(split_work_clip, step, scale_max, scale_min)

    full_clip = add_movie([split_begin_clip, split_work_clip_ret, split_end_clip])

    return full_clip


# 放大 平移
# duration 动画时长
# from_scale 起点放大倍数
# end_scale 终点放大倍数
# zoom中心 在图片的中点
# from_xy 起始点中心坐标
# end_xy  结束点中心坐标
def make_frame_scale_translate(duration, from_scale, end_scale, from_xy, end_xy):
    def frame_scale(gf, t):
        frame = gf(t)
        original_height, original_width, _ = frame.shape

        scale_cha = end_scale - from_scale
        scale_frame = from_scale + scale_cha * (t / duration)
        new_width = int(frame.shape[1] * scale_frame)
        new_height = int(frame.shape[0] * scale_frame)
        # 使用 OpenCV 的 resize 函数和插值方法放大图像
        zoomed_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        from_xy_np = np.array(from_xy)
        end_xy_np = np.array(end_xy)

        cha_xy_np = end_xy_np - from_xy_np
        # 计算当前平移位置
        res_xy_np = from_xy_np + cha_xy_np * (t / duration)
        # 转为整数
        res_xy_np = res_xy_np.astype(np.int32)

        # 计算裁剪的起始位置 先计算放缩影响 然后计算平移影响
        start_x = (new_width - original_width) // 2 - res_xy_np[0]
        start_y = (new_height - original_height) // 2 - res_xy_np[1]

        if start_x < 0:
            start_x = 0
        if start_x > new_width - original_width:
            start_x = new_width - original_width

        if start_y < 0:
            start_y = 0
        if start_y > new_height - original_height:
            start_y = new_height - original_height

        # 裁剪放大后的图像，使其与原始图像的尺寸相同
        zoomed_frame = zoomed_frame[start_y:start_y + original_height, start_x:start_x + original_width]

        # print("res_xy_np:", res_xy_np)
        # print("start_x:", start_x, "start_y", start_y)
        # print("--------------------")
        #
        # cv2.imshow("orgin", frame)
        # cv2.imshow("scale", zoomed_frame)
        # cv2.waitKey()

        return zoomed_frame

    return frame_scale


def give_me_rand_xy(video_clip, scale_max):
    width, height = video_clip.size
    size = np.array([width, height])
    max_size = size * scale_max
    max_xy = (max_size - size) // 2
    min_xy = - max_xy
    np_random = np.random.uniform(low=min_xy, high=max_xy, size=max_size.shape).astype(np.int32)
    return np_random


def movie_curve_change_scale_work_step(video_clip, scale_max):
    movie_time = get_movie_time(video_clip)
    duration = movie_time / 2
    split_1_clip = movie_split_by_time(video_clip, 0, duration)
    split_2_clip = movie_split_by_time(video_clip, duration, movie_time)

    end_xy = give_me_rand_xy(video_clip, scale_max)

    new_1_clip = split_1_clip.fl(make_frame_scale_translate(duration, 1, scale_max, [0, 0], end_xy))

    new_2_clip = split_2_clip.fl(make_frame_scale_translate(duration, scale_max, 1, end_xy, [0, 0]))

    return add_movie([new_1_clip, new_2_clip])


def movie_curve_change_scale_work(video_clip, step, scale_max):
    move_time = get_movie_time(video_clip)
    ret_clips = []
    start_time = 0
    while start_time < move_time:
        end_time = start_time + step
        # 确保结束时间不超过视频总时长
        if end_time > move_time:
            end_time = move_time
        subclip = movie_split_by_time(video_clip, start_time, end_time)
        subclip = movie_curve_change_scale_work_step(subclip, scale_max)
        ret_clips.append(subclip)
        # 保存分割后的视频片段
        # 更新起始时间
        start_time += step

    return add_movie(ret_clips)


def movie_curve_change_scale(video_clip, start, end, step, scale_max):
    movie_time = get_movie_time(video_clip)
    # 开始和结束clip 不动 最后进行拼接 中间作业的clip 传入另一个方法中进行剪辑
    split_begin_clip = movie_split_by_time(video_clip, 0, start)
    split_end_clip = movie_split_by_time(video_clip, end, movie_time)
    split_work_clip = movie_split_by_time(video_clip, start, end)

    split_work_clip_ret = movie_curve_change_scale_work(split_work_clip, step, scale_max)

    full_clip = add_movie([split_begin_clip, split_work_clip_ret, split_end_clip])

    return full_clip
