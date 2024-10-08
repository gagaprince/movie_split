from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import cv2
import random


def parse_clip(in_file):
    return VideoFileClip(in_file)


def save_clip(clip, out_file):
    clip.write_videofile(out_file, codec='libx264', audio_codec='aac')


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


def give_me_rand_xy(video_clip, scale_max, origin_xy=np.array([0, 0])):
    width, height = video_clip.size
    size = np.array([width, height])
    max_size = size * scale_max
    max_xy = (max_size - size) // 2
    min_xy = - max_xy
    np_random = np.random.uniform(low=min_xy, high=max_xy, size=max_size.shape).astype(np.int32)
    # if np.linalg.norm(np_random - origin_xy) > width/5:
    return np_random
    # else:
    #     return give_me_rand_xy(video_clip, scale_max, origin_xy)


def movie_curve_change_scale_work_auto_step(video_clip, scale_max, from_scale=1, from_xy=None):
    if from_xy is None:
        from_xy = [0, 0]
    movie_time = get_movie_time(video_clip)
    duration = movie_time / 2
    split_1_clip = movie_split_by_time(video_clip, 0, duration)
    split_2_clip = movie_split_by_time(video_clip, duration, movie_time)

    end_xy = give_me_rand_xy(video_clip, scale_max)

    scale_max = random.uniform(1.2, scale_max)
    # ret_scale = max(random.uniform(1, scale_max), 1)
    # ret_xy = give_me_rand_xy(video_clip, ret_scale, end_xy)

    ret_scale = 1
    ret_xy = [0, 0]

    new_1_clip = split_1_clip.fl(make_frame_scale_translate(duration, from_scale, scale_max, from_xy, end_xy))

    new_2_clip = split_2_clip.fl(make_frame_scale_translate(duration, scale_max, ret_scale, end_xy, ret_xy))

    return [add_movie([new_1_clip, new_2_clip]), ret_scale, ret_xy]


# 自动切分 自动放缩 自动平移
def movie_curve_change_scale_auto_work(video_clip, step, scale_max):
    move_time = get_movie_time(video_clip)
    ret_clips = []
    start_time = 0
    ret_scale = 1
    ret_xy = [0, 0]
    while start_time < move_time:
        end_time = start_time + step
        # 确保结束时间不超过视频总时长
        if end_time > move_time:
            end_time = move_time
        subclip = movie_split_by_time(video_clip, start_time, end_time)
        subclip, ret_scale, ret_xy = movie_curve_change_scale_work_auto_step(subclip, scale_max, ret_scale, ret_xy)
        ret_clips.append(subclip)
        # 保存分割后的视频片段
        # 更新起始时间
        start_time += step

    return add_movie(ret_clips)


def movie_curve_change_auto_scale(video_clip, start, end, step, scale_max):
    movie_time = get_movie_time(video_clip)
    # 开始和结束clip 不动 最后进行拼接 中间作业的clip 传入另一个方法中进行剪辑
    split_begin_clip = movie_split_by_time(video_clip, 0, start)
    split_end_clip = movie_split_by_time(video_clip, end, movie_time)
    split_work_clip = movie_split_by_time(video_clip, start, end)

    split_work_clip_ret = movie_curve_change_scale_auto_work(split_work_clip, step, scale_max)

    full_clip = add_movie([split_begin_clip, split_work_clip_ret, split_end_clip])

    return full_clip


def movie_curve_change_scale_work_step(video_clip, cut_map_item):
    _, scale_obj, pointxy_obj = cut_map_item
    movie_time = get_movie_time(video_clip)
    width, height = video_clip.size
    from_xy = [pointxy_obj[0][0] * width, pointxy_obj[0][1] * height]
    end_xy = [pointxy_obj[1][0] * width, pointxy_obj[1][1] * height]
    new_clip = video_clip.fl(make_frame_scale_translate(movie_time, scale_obj[0], scale_obj[1], from_xy, end_xy))
    return new_clip


"""
按照编好的 地图完成剪辑
地图的格式：[时间段] [起始scale，结束scale] [起始位置， 结束位置]  对应的是切片 + 切片起点到终点关键帧的动画
[
    [
        [t1, t2],[scale1, scale2], [[x1, y1], [x2, y2]]
    ],
    [
        [t2, t3],[scale2, scale3], [[x2, y2], [x3, y3]]
    ],
    [
        [t3, t4],[scale3, scale4], [[x3, y3], [x4, y4]]
    ]
]
"""
def movie_curve_change_scale(video_clip, cut_map):
    movie_time = get_movie_time(video_clip)

    full_split_clips = []

    for split_obj in cut_map:
        # 拿到时间点
        start, end = split_obj[0]
        if start > movie_time:
            raise
        if end > movie_time:
            end = movie_time
        split_clip = movie_split_by_time(video_clip, start, end)
        split_clip = movie_curve_change_scale_work_step(split_clip, split_obj)
        full_split_clips.append(split_clip)

    return add_movie(full_split_clips)


def old_film(image):
    # 添加噪点和颜色偏移
    noise = np.random.randint(0, 50, (image.shape[0], image.shape[1], 3))
    image = image + noise
    image[:, :, 0] = np.clip(image[:, :, 0] * 1.2, 0, 255)  # 增加红色
    image[:, :, 1] = np.clip(image[:, :, 1] * 0.8, 0, 255)  # 减少绿色
    image[:, :, 2] = np.clip(image[:, :, 2] * 0.8, 0, 255)  # 减少蓝色
    return image


def movie_old_film(video_clip):
    return video_clip.fl_image(old_film)
