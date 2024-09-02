import cv2
import os
import numpy as np
# from skimage.metrics import structural_similarity as compare_ssim


def extract_frames(video_path, output_folder, fps=1):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每隔多少帧提取一帧
    frame_interval = int(video_fps / fps)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 frame_interval 帧提取一帧
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f'frame_{extracted_count:04d}.jpg')
            cv2.imwrite(output_path, frame)
            extracted_count += 1
            print('save 1pic in', output_path)

        frame_count += 1

    # 释放视频捕获对象
    cap.release()


# def align_images(image1, image2):
#     # 将图像转换为灰度图像
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#
#     # 使用光流法估计位移
#     flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#     # 获取位移的平均值
#     print('flow:', flow)
#     dx = np.mean(flow[..., 0])
#     dy = np.mean(flow[..., 1])
#
#     print('dx:', dx, ' dy:', dy)
#
#     # 构建平移矩阵
#     translation_matrix = np.float32([[1, 0, -30*dx], [0, 1, -30*dy]])
#
#     # 对图像进行平移校正
#     aligned_image2 = cv2.warpAffine(image2, translation_matrix, (image2.shape[1], image2.shape[0]))
#
#     return aligned_image2


def crop_center(image, crop_size_x=None, crop_size_y=None):
    h, w = image.shape[:2]
    if crop_size_x is None:
        crop_size_x = w // 2
    if crop_size_y is None:
        crop_size_y = h // 2
    start_x = w // 2 - crop_size_x // 2
    start_y = h // 2 - crop_size_y // 2
    return image[start_y:start_y + crop_size_y, start_x:start_x + crop_size_x]


def compare_frames(frame1, frame2):
    # 对齐两帧
    # aligned_frame2 = align_images(frame1, frame2)

    # 将帧转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 截取图像中心的9宫格区域
    cropped_gray1 = crop_center(gray1)
    cropped_gray2 = crop_center(gray2)

    # 计算两帧之间的绝对差异
    diff = cv2.absdiff(cropped_gray1, cropped_gray2)

    # 计算差异的总和
    diff_sum = np.sum(diff)
    return diff_sum


def search_change_frame_and_save(video_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()

    index = 0
    ret_list = []

    while True:
        # 读取下一帧
        ret, frame2 = cap.read()
        if not ret:
            break

        # 比较两帧的差异
        diff = compare_frames(frame1, frame2)
        output_path = os.path.join(output_folder, f'frame_{index:06d}.jpg')
        # cv2.imwrite(output_path, frame1)
        ret_list.append({
            "diff": diff,
            "output_path": output_path,
            "index": index
        })
        # 更新帧
        frame1 = frame2
        index += 1

    select_list = []

    length = len(ret_list)
    for idx in range(length):
        if idx < length - 1:
            pre_frame_info = ret_list[idx]
            next_frame_info = ret_list[idx + 1]
            if pre_frame_info['diff'] / (next_frame_info['diff'] + 1) < 0.3:
                select_list.append(next_frame_info)
    select_list.append(ret_list[length - 1])
    print('查找到的帧:', select_list)

    for item in select_list:
        frame_number = item['index']
        output_path = item['output_path']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # 读取指定帧
        ret, frame = cap.read()
        if ret:
            # 保存当前帧为图片
            cv2.imwrite(output_path, frame)
            print(f"已保存第 {frame_number} 帧为图片: {output_path}")
        else:
            print(f"无法读取第 {frame_number} 帧")

    # 释放视频捕获对象
    cap.release()


if __name__ == '__main__':
    video_path = './movie/test.mp4'
    output_folder = './imgs/tmp2/'
    search_change_frame_and_save(video_path, output_folder)


