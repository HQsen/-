import cv2
import numpy as np
import os
import tqdm
import mediapipe as mp
from FBPE import FullBodyPoseEmbedder
from PC import PoseClassifier
from EMA import EMADictSmoothing
from RC import RepetitionCounter

#样本所比较类别
class_name='up'
'''初始化'''
#指定所读取的csv文件夹
pose_samples_folder = 'squat_csvs_out'

#初始化人体姿态跟踪器
pose_tracker = mp.solutions.pose.Pose(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)

 #初始化embedder，准备所要嵌入的17个节点.
pose_embedder=FullBodyPoseEmbedder()

#初始化分类器.
'''读取csv文件和其中每张图的17个关键点每个点的xy坐标（54个数）、对应类别、图片路径，对识别样本进行分类'''
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removing them): ', len(outliers))

#初始化EMA平滑
'''姿态分类结果平滑，平滑置信度（移动平均）'''
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

#指定动作阈值.
'''获取置信度并对比，服务于动作计数'''
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=5.8,#enter_threshold：置信度需要超过的下限
    exit_threshold=5)#exit_threshold：回落值，回落后计数+1

width = 480
height = 640
landmarks=[{'name':'left_leg','x':156,'y':512},{'name':'left_leg','x':156,'y':512},{'name':'left_leg','x':156,'y':512},{'name':'left_leg','x':156,'y':512},{'name':'left_leg','x':156,'y':512}]
result=[]
for i in landmarks:
    tem=[]
    for k,v in i.items():
        tem.append(v)
    result.append(tem[1:])
result=np.array(result)
if result is not None:
    pose_landmarks = np.array([[lmk.x * width, lmk.y * height]
                    for lmk in result], dtype=np.float32)  # 根据图片尺寸放大关键点
    assert pose_landmarks.shape == (17, 2), 'Unexpected landmarks shape: {}'.format(
        pose_landmarks.shape)  # 如果检测不到尺寸，设置默认尺寸
    # 姿势分类
    pose_classification = pose_classifier(pose_landmarks)
    # EMA平滑分类
    pose_classification_filtered = pose_classification_filter(pose_classification)

    # 返回T or F
    true_or_false = repetition_counter(pose_classification_filtered)
else:
    # No pose => no classification on current frame.
    pose_classification = None

    # Still add empty classification to the filter to maintaing correct
    # smoothing for future frames.
    pose_classification_filtered = pose_classification_filter(dict())
    pose_classification_filtered = None

    # Don't update the counter presuming that person is 'frozen'. Just
    # take the latest repetitions count.
    repetitions_count = False
print(true_or_false)


# #指定处理好视频的路径和名称
# video_path = 'video.mp4'
# out_video_path = 'video-output.mp4'

# # 获取摄像头，传入0表示获取系统默认摄像头（Windows）,1为外接摄像头（Mac）
# cap = cv2.VideoCapture(0)
# # 打开cap
# cap.open(0)
# frame_idx = 0
# output_frame = None
# while cap.isOpened():
#     # 获取画面
#     success, input_frame= cap.read()# 获取读取状态和当前帧
#     if not success:
#         break
#
#     # 转RGB,把图输入模型获取预测结果
#     input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
#     result = pose_tracker.process(image=input_frame)
#     pose_landmarks = result.pose_landmarks  # 关键点
#
#     # 绘制动作预测
#     output_frame = input_frame.copy()  # 复制
#     if pose_landmarks is not None:
#         mp.solutions.drawing_utils.draw_landmarks(
#             image=output_frame,
#             landmark_list=pose_landmarks,
#             connections=mp.solutions.pose.POSE_CONNECTIONS)  # 绘图
#         # Get landmarks.
#         frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]  # 读取图片长宽
#         pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height]
#                                    for lmk in pose_landmarks.landmark], dtype=np.float32)  # 根据图片尺寸放大关键点
#         assert pose_landmarks.shape == (17, 2), 'Unexpected landmarks shape: {}'.format(
#             pose_landmarks.shape)  # 如果检测不到尺寸，设置默认尺寸
#
#         # 姿势分类
#         pose_classification = pose_classifier(pose_landmarks)
#
#         # EMA平滑分类
#         pose_classification_filtered = pose_classification_filter(pose_classification)
#
#         # 返回T or F
#         true_or_false = repetition_counter(pose_classification_filtered)
#     else:
#         # No pose => no classification on current frame.
#         pose_classification = None
#
#         # Still add empty classification to the filter to maintaing correct
#         # smoothing for future frames.
#         pose_classification_filtered = pose_classification_filter(dict())
#         pose_classification_filtered = None
#
#         # Don't update the counter presuming that person is 'frozen'. Just
#         # take the latest repetitions count.
#         repetitions_count = repetition_counter.n_repeats
#     output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
#     # 展示处理后的三通道图像
#     cv2.imshow('my_window', output_frame)
#     # 输出计数
#     print(true_or_false)
#     frame_idx += 1  # 帧数
#     if frame_idx==3600 or cv2.waitKey(1)==27:  #这边暂时以帧数计时，计时一分钟退出
#         break
# # 关闭摄像头
# cap.release()
# # 关闭图像窗口
# cv2.destroyAllWindows()

