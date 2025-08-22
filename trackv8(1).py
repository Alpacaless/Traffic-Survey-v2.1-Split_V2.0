import sys
sys.path.insert(0, './yolov5')
# from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import myLib.get_PID_and_Kill
import supervision as sv
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.downloads import attempt_download
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                  colorstr, increment_path, non_max_suppression, print_args, scale_boxes,
                                  strip_optimizer)
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from tools import ToolVehicle
from random import randint
import argparse,platform,time, datetime,re
from datetime import datetime, timedelta
from pathlib import Path
import cv2,os,torch,threading,pathlib
import torch.backends.cudnn as cudnn
# from torchvision.transforms import Equalize
import requests, json, base64
from requests.exceptions import RequestException
from flask import Flask, request, jsonify
import numpy as np
from random import randint
from deep_sort_pytorch.deep_sort import DeepSort
from datetime import datetime,timedelta,date
import subprocess
import AutoReCheckTime,RebootCheckTime
import shutil


# 全局运行线程控制变量
current_run_thread = None
stop_run_flag = False


plt = platform.system()
if plt != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath
import os

def exit_program():
    print("退出程序，终止线程")
    os._exit(0)  # 立即强制退出整个Python程序（包括所有线程）




total_num = 0  # 初始化车辆总数
RUN_SWITCH = False  # 运行开关，用于控制任务执行
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)  # 调色板，用于绘制检测框

# todo
# 检测线的两个端点坐标，表示车辆经过时的检测区域
# line = [500, 540, 1860, 540]  # 检测线的两个端点的xy坐标，总共4个数  左边（x1,y1）  右边（x2, y2）
# line = [770, 375, 1890, 375]  # 检测线的两个端点的xy坐标，总共4个数  左边（x1,y1）  右边（x2, y2）
line = [250, 350, 1850, 350]  # 检测线的两个端点的xy坐标，总共4个数  左边（x1,y1）  右边（x2, y2）
# line = [int(250/1.5), int(350/1.5), int(1850/1.5), int(350/1.5)]


# 定义设备编号
# device_code1 = '0071154316110261'       # 会泽1
# device_code1 = '9991180324070008'       # 会泽2
# device_code1 = '9991180324070001'       # 保山1
# device_code1 = '9991180324070002'       # 保山2
# device_code1 = '9991180324070003'       # 保山3，7
# device_code1 = '9991180324070004'       # 保山4
# device_code1 = '9991180324070005'       # 保山5
# device_code1 = '9991180324070006'       # 保山6
device_code1 = '0021145319062069'       # 泸西
# device_code1 = '0171170315123001'       # 马龙
# device_code1 = '28254'  # 测试

import json
from datetime import datetime

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def parse_time(time_str):
    """处理24小时制时间字符串"""
    if time_str == "24:00:00":
        return datetime.strptime("23:59:59", "%H:%M:%S").time()
    return datetime.strptime(time_str, "%H:%M:%S").time()

def get_current_config():
    configs = load_config()
    now = datetime.now().time()

    for config in configs['configurations']:
        start_str, end_str = config['time_range'].split('-')
        start_time = parse_time(start_str)
        end_time = parse_time(end_str)

        # 处理跨午夜的时间段
        if start_time < end_time:
            if start_time <= now <= end_time:
                return config['source'], config['weights'], config['save_dir']
        else:  # 跨午夜的时间段
            if now >= start_time or now <= end_time:
                return config['source'], config['weights'], config['save_dir']

    # 默认配置
    # ✅ 加默认配置兜底，避免返回 None 报错
    print("⚠️ 当前未命中任何配置，使用默认配置")
    return (
        "rtsp://admin:admin12345@192.168.10.51/cam/realmonitor?channel=1&subtype=0",
        "yolov5/weights/day.pt",
        "inference/output"
    )

url1 = f"http://222.219.137.122:19030/api/device/{device_code1}"  # 设备状态请求地址
headers1 = {'Content-Type': 'application/x-www-form-urlencoded'}  # 请求头信息
# 数据上传地址
upload_addr3 = 'http://222.219.137.122:19030/api/analyze-result'  # 车辆信息上传地址
CLIENT_PORT = '6060'  # 客户端端口号

# 车辆类型映射表，用于标记上传数据中的车辆类型
uploadNameMap = {"Motorcycle": 'Motorcycle', "Car": 'MidsizeCar', "Bus": 'LargeBus', "Tractor": 'Tractor',
                 "L_truck": 'SmallTruck', "XL_truck": 'MediumTruck', "XXL_truck": 'LargeTruck',
                 "XXXL_truck": 'OversizeTruck', "Container car": 'ContainerTruck', "Electric vehicle": 0, "Total": 0}

#测速初始化
FPS=10
SOURCE = np.array([[812, 189], [1075, 188], [852, 1080], [-431, 1080]])
LINE_start=(SOURCE[0]+SOURCE[3])/2
LINE_end=(SOURCE[1]+SOURCE[2])/2
start, end = sv.Point(x=0, y=int(LINE_start[1])), sv.Point(x=1920, y=int(LINE_end[1]))
TARGET_WIDTH = 6
TARGET_HEIGHT = 80

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

error_dir = "inference/error_logger"  # 错误日志文件保存目录
os.makedirs(error_dir, exist_ok=True)  # 创建错误日志目录

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# 定义函数，将目标框的绝对坐标转换为相对坐标（中心点、宽、高）
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    # x_c = (xyxy[0]+ xyxy[2])/2
    y_c = (bbox_top + bbox_h / 2)
    # y_c = (xyxy[1]+ xyxy[3])/2
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h  # 返回中心点、宽度和高度

# 将xyxy坐标转换为tlwh格式（左、上、宽、高），便于后续处理
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]  # 提取每个框的左上角和右下角坐标
        top = x1  # 顶部为x1
        left = y1  # 左侧为y1
        w = int(x2 - x1)  # 计算宽度
        h = int(y2 - y1)  # 计算高度
        tlwh_obj = [top, left, w, h]  # 构造tlwh格式的边界框
        tlwh_bboxs.append(tlwh_obj)  # 添加到列表
    return tlwh_bboxs  # 返回边界框列表

def EqualizeHistRGB(imSource,model):
    imSource = np.squeeze(imSource)
    rh = cv2.equalizeHist(imSource[0, :, :])
    gh = cv2.equalizeHist(imSource[1, :, :])
    bh = cv2.equalizeHist(imSource[2, :, :])
    imMerge = cv2.merge((rh, gh, bh), )
    imMerge = imMerge.transpose((2, 0, 1));

    return im

import random
# 全局变量：已上传文件的路径存储
uploaded_files = []
upload_count = 0  # 文件上传计数器
error_dir = "inference/error_logger"  # 替换为实际的错误日志目录
last_save_time = None
def WriteImgAndTxt(Im,img_path,txt_path,upload_dic,txt_name, current_folder):
    global upload_count, uploaded_files, last_save_time

    date_time_h = datetime.now().strftime('%Y-%m-%d_%H')
    error_name = f'{date_time_h}_error_logg.txt'
    error_path = os.path.join(error_dir, error_name)
    # cv2.imwrite(img_path, Im)  # 保存图片

    # 检查上次保存的时间和当前时间的间隔
    current_time = datetime.now()
    if is_within_time_range():
        if last_save_time is None or (current_time - last_save_time).total_seconds() >= 30:
            cv2.imwrite(img_path, Im)  # 保存图片
            print(f"保存图片:{os.path.basename(img_path)}")
            last_save_time = current_time  # 更新保存时间
    else:
        if last_save_time is None or (current_time - last_save_time).total_seconds() >= 3600:
            cv2.imwrite(img_path, Im)  # 保存图片
            print(f"保存图片:{os.path.basename(img_path)}")
            last_save_time = current_time  # 更新保存时间

    with open(txt_path, 'a') as f:
        json_string = json.dumps(upload_dic)
        f.write(json_string + '\n')

    # 上传文件
    if send_file_to_server(txt_path):
        now_time = datetime.now()
        uploaded_files.append((txt_path, now_time))  # 记录上传文件路径及其时间戳
        upload_count += 1

        # 每上传20个文件，从当前文件夹中的后10%文件中随机选择一个重新上传
        if is_within_time_range():
            if upload_count % 18 == 0:
                reupload_from_last_10_percent(current_folder)
            if upload_count % 50 == 0:
                resend_midsizecar(current_folder)
    else:
        log_error(txt_name)

    return 0

def is_within_time_range():
    """检查当前时间是否在18点前"""
    current_time = datetime.now().time()
    return  current_time.hour < 19 and current_time.hour > 9  # 小时数小于18表示在18点前

def reupload_from_last_10_percent(current_folder):
    try:
        # 获取当前文件夹中的所有文件并按修改时间排序
        all_files = [
            os.path.join(current_folder,f) for f in os.listdir(current_folder)
            if os.path.isfile(os.path.join(current_folder, f))
        ]
        all_files.sort(key=os.path.getmtime)  # 按文件修改时间排序

        # 计算后10%的文件范围
        total_files = len(all_files)
        if total_files == 0:
            print("当前文件夹没有文件可供选择。")
            return

        last_10_percent_count = max(1, total_files // 10)  # 至少选一个
        candidates = all_files[-last_10_percent_count:]

        # 随机选择一个文件重新上传
        random_file = random.choice(candidates)
        update_and_reupload(random_file)
    except Exception as e:
        print(f"重传错误: {e}")

def resend_midsizecar(current_folder):
    try:
        # 获取当前文件夹中的所有文件并按修改时间排序
        all_files = [
            os.path.join(current_folder, f) for f in os.listdir(current_folder)
            if os.path.isfile(os.path.join(current_folder, f))
        ]
        all_files.sort(key=os.path.getmtime)  # 按文件修改时间排序

        # 计算后10%的文件范围
        total_files = len(all_files)
        if total_files == 0:
            print("当前文件夹没有文件可供选择。")
            return

        last_10_percent_count = max(1, total_files // 10)  # 至少选一个
        candidates = all_files[-last_10_percent_count:]

        # 筛选midsizecar和smalltruck的文件
        filtered_files = []
        for file_path in candidates:
            try:
                with open(file_path, 'r') as file:
                    content = json.load(file)
                    vehicle_model = content.get('vehicleModel', '').lower()
                    if vehicle_model in ['midsizecar', 'smalltruck']:
                        filtered_files.append(file_path)
            except Exception as e:
                print(f"无法处理文件 {file_path}: {e}")

        if not filtered_files:
            print("未找到符合条件的文件。")
            return

        # 随机选择一个文件上传
        random_file = random.choice(filtered_files)
        update_and_reupload(random_file)

    except Exception as e:
        print(f"重新上传时发生错误: {e}")


def update_and_reupload(file_path):
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"重传文件: {file_path}，更新为当前时间：{now_time}")

    # 修改文件中的时间戳
    with open(file_path, 'r') as f:
        content = f.read()

    try:
        # 使用正则提取第一个JSON对象
        match = re.match(r'({.*?})', content, re.DOTALL)
        if not match:
            print(f"文件 {file_path} 不包含有效的JSON对象。")
            return
        # 提取第一个JSON对象
        upload_dic = json.loads(match.group(1))
        # upload_dic = json.loads(content)
        upload_dic['detectionTime'] = now_time  # 更新为当前时间

        with open(file_path, 'w') as f:
            json.dump(upload_dic, f)

        # 重新上传文件
        send_file_to_server(file_path)
    except json.JSONDecodeError as e:
        print(f"文件内容格式错误，无法解析为JSON: {e}")
    except Exception as e:
        print(f"更新文件时发生错误: {e}")


def log_error(txt_name):
    date_time_h = datetime.now().strftime('%Y-%m-%d_%H')
    error_name = f'{date_time_h}_error_logg.txt'
    error_path = os.path.join(error_dir, error_name)

    with open(error_path, "a") as error_file:
        error_file.write(txt_name + "\n")

import torchvision.transforms.functional as F
def torchvision_histogram_equalization(Im,model):
    Im = np.squeeze(Im)
    Im = torch.from_numpy(Im).to(model.device)  # 将numpy数组转换为PyTorch张量，并移动到指定设备
    # 将图像分成单通道，并对每个通道均衡化
    r, g, b = Im.split(1)  # 分割成单通道的图像
    r = F.equalize(r)  # 对红色通道进行均衡化
    g = F.equalize(g)  # 对绿色通道进行均衡化
    b = F.equalize(b)  # 对蓝色通道进行均衡化
    Im_balanced = torch.cat([r, g, b], dim=0)   # 将均衡化后的通道合并
    return Im_balanced

def Video_save(im0,vid_cap,vid_path,save_path,vid_writer):
    # imResize=cv2.resize(im0,(1280,720))
    imResize = im0
    if vid_path != save_path:  # 如果是新的视频，创建新的视频写入器
        vid_path = save_path
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # 释放之前的视频写入器
        if vid_cap:  # 如果是视频文件，获取FPS和视频尺寸
            fps, w, h = vid_cap.get(cv2.CAP_PROP_FPS), imResize.shape[1], imResize.shape[0]
            #fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的FPS
            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
            # w = 1280
            # h = 720
        else:  # 如果是流媒体，设定固定的FPS和尺寸
            fps, w, h = 10, imResize.shape[1], imResize.shape[0]
            save_path += '.mp4'  # 强制保存为mp4格式
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入器
    vid_writer.write(imResize)  # 写入帧到视频文件


def video_savev2(vid_cap, im0, save_path):
    global vid_writer
    global vid_path
    if vid_path != save_path:  # 如果是新的视频，创建新的视频写入器
        vid_path = save_path
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # 释放之前的视频写入器
        if vid_cap:  # 如果是视频文件，获取FPS和视频尺寸
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
            # w = 1500  # 需要修改成裁剪后的尺寸。 如果是原视频就用上面一行。
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
        else:  # 如果是流媒体，设定固定的FPS和尺寸
            fps, w, h = 15, im0.shape[1], im0.shape[0]
            save_path += '.mp4'  # 强制保存为mp4格式
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入器
        im1 = cv2.resize(im0, (720, 1280))
    vid_writer.write(im1)  # 写入帧到视频文件


# 启用智能推理模式
@smart_inference_mode()
def run(weights='yolov5/weights/yolov5s.pt',  # model path or triton URL
        deep_sort_weights='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt1.t7',
        source='0',  # file/dir/URL/glob/screen/0(webcam)
        data='yolov5/data/coco128.yaml',  # dataset.yaml path
        imgsz=[500],  # inference size (height, width)q
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device=0,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save=True,  # save images/videos
        ref_time='2023-01-01 00:00:00',  # '%Y-%m-%d %H:%M:%S'
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        config_deepsort='deep_sort_pytorch/configs/deep_sort.yaml',
        camera_device='192.168.10.3',
        ):
    cfg = get_config()  # 加载DeepSort的配置
    lost_frame_count = 0  # 初始化丢帧计数器
    MAX_LOST_FRAMES = 30  # 连续丢帧30次则重启，大约30秒
    global stop_run_flag
    cfg.merge_from_file(config_deepsort)  # 将配置文件与默认配置合并
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')  # 下载DeepSort权重文件
    # deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,  # 使用配置初始化DeepSort
    #                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    #                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    #                     percent=cfg.DEEPSORT.PERCENT, use_cuda=True)  # 设置DeepSort参数
    # 初始化 ByteTrack
    bytetrack= sv.ByteTrack(track_activation_threshold=0.25,lost_track_buffer=FPS,minimum_matching_threshold=0.8,frame_rate=FPS,minimum_consecutive_frames=1)
    
    # 初始化上传的字典，包含设备编码、图像等信息
    uploadBaseDic = {"deviceCode": '28254', "image": '0', "vehicleModel": '0', "speed": 0, "lanesNumber": 0,
                     "detectionTime": '0'}
    uploadBaseDic['deviceCode'] = camera_device

    # 初始化工具类 ToolVehicle

    #测速
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    line_zone = sv.LineZone(start=start, end=end)
    tool_vehicle = ToolVehicle(line, FPS)  # 用于统计车辆信息

    # picture_num = ToolVehicle(total_num)  # 统计总数
    source = str(source)  # 转换为字符串类型
    save_img = save and (not source.endswith('.txt'))  # 判断是否保存推理后的图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 检查是否为文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断是否为URL
    # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    webcam = source.isnumeric() or source.endswith('.txt') or is_url  # 检查是否为摄像头或流媒体
    screenshot = source.lower().startswith('screen')  # 检查是否为截图

    # if is_url and is_file:
    #    source = check_file(source)  # download

    # http开头, .mp4结尾为录播，  http开头, .flv结尾为直播
    # playback = (source.startswith('http') and source.endswith('.mp4'))  # 数据来源
    playback = (source.startswith('http') and source.endswith('.flv'))  # 判断是否为直播流

    # 创建保存结果的目录
    if save_img:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 自动递增路径
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建结果保存目录
        number_path = str(save_dir / 'number.txt')  # 计数信息保存位置

    # 加载模型
    device = select_device(device)  # 选择设备（GPU或CPU）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 初始化YOLO模型
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型步长、类别名称和模型类型
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小是否符合模型要求
    print("输出检测类别：", names)  # 打印检测到的类别

    # Dataloader数据加载器，用于加载视频或图像流
    bs = 1  # batch_size
    if webcam:  # 如果是摄像头或流媒体
        cudnn.benchmark = True  # 设置为True加速固定图像大小的推理
        check_imshow(warn=True)  # 检查是否支持imshow
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载流媒体数据
        bs = len(dataset)  # 更新batch_size
    elif screenshot:  # 如果是截图
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # 加载截图数据
    else:  # 加载本地视频或图像
        cudnn.benchmark = True  # 设置为True加速固定图像大小的推理
        check_imshow(warn=True)  # 检查是否支持imshow
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载图像数据
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化视频写入器

    # 初始化上传数据
    total_count = 0
    down_detail = {"Motorcycle": 0, "Car": 0, "Bus": 0, "Tractor": 0, "L_truck": 0, "XL_truck": 0, "XXL_truck": 0,
                   "XXXL_truck": 0, "Container car": 0, "Electric vehicle": 0, "Total": 0}
    up_detail = {"Motorcycle": 0, "Car": 0, "Bus": 0, "Tractor": 0, "L_truck": 0, "XL_truck": 0, "XXL_truck": 0,
                 "XXXL_truck": 0, "Container car": 0, "Electric vehicle": 0, "Total": 0}
    trackInfoList = []  # 跟踪信息列表

    headers = {'Content-Type': 'application/json'}

    # http录播状态时，给前端发送“开始分析”信号
    if playback:
        upload_address = upload_addr3 + '/' + source.split('/')[-1] + '/' + 'InProcess'
        r = requests.get(upload_addr3, headers=headers, params=json.dumps({'fileName': source.split('/')[-1], 'status': 'InProcess'}))

        # 获得回放视频的历史开始时间戳，用于叠加分析时间
        ref_time = datetime.strptime(ref_time, '%Y-%m-%d %H:%M:%S')
        ref_time = datetime.timestamp(ref_time)

    # Run inference，加速推理速度
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, deeosortNum, windows, dt = 0, 0, [], (Profile(), Profile(), Profile(), Profile())  # 初始化计时器
    t0 = time.time()  # 开始计时
    DataNum, CarTotal = 0, 0
    old_time = datetime.now().strftime('%Y-%m-%d')
    # labels = defaultdict(lambda:np.array([0]))
    # 遍历每一帧图像，执行推理和跟踪
    for frame_idx, (path, im, im0s, vid_cap, timestamp, ret_flag) in enumerate(dataset):

        # print(f"[DEBUG] 正在处理帧 {frame_idx}")
        # 新增退出标志检查
        if stop_run_flag:
            print("收到中止标志，run() 正在退出...")
            break
        # print("裁剪后的图像shape:", img.shape)
        # ---------------------设置区域检测的范围  如果不需要显示，可以在下面注释掉--------------------------
        # Detect.region_detect(im, webcam)
        # ---------------------------------------------------------------------------------------------------------------
        if not ret_flag:
            lost_frame_count += 1
            LOGGER.warning(f"第 {lost_frame_count} 次未成功读取帧")
            if lost_frame_count >= MAX_LOST_FRAMES:
                LOGGER.error("连续丢帧达到上限，准备退出程序！")
                stop_run_flag = True  # 标记退出
                exit_program()
                break
            time.sleep(1)
            continue
        else:
            lost_frame_count = 0  # 成功读取帧，重置计数器

        # 处理图像数据，转换为模型可用的格式
        with dt[0]:
            # im=EqualizeHistRGB(im,model) # CUP增强
            # im = torchvision_histogram_equalization(im,model) # GPU增强
            im = torch.from_numpy(im).to(model.device)  # 将numpy数组转换为PyTorch张量，并移动到指定设备
            im = im.half() if model.fp16 else im.float()  # 根据模型是否使用FP16，选择半精度或浮点精度
            im /= 255  # 将像素值0 - 255归一化到0.0 - 1.0之间
            if len(im.shape) == 3:
                im = im[None]  # 扩展维度，适应批处理
        s = ""  # 初始化为一个空字符串
        # 推理阶段，使用YOLOv5模型进行目标检测
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 处理可视化路径
            pred = model(im, augment=augment, visualize=visualize)  # 执行推理，获得预测结果

        # 非极大值抑制，去除多余的检测框
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 处理预测结果
        with dt[3]:
            # 遍历每张图片的预测结果
            for i, det in enumerate(pred):
                seen += 1  # 增加处理帧的计数

                if webcam:  # 如果是摄像头输入，逐帧处理
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    # s += f'{i}: '
                    # ---------------------画出区域检测的范围  如果不需要显示，可以在下面注释掉--------------------------
                    # Detect.show_region_detection(im0)
                    # ---------------------------------------------------------------------------------------------------
                else:  # 如果是文件或视频流
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    # p, im0, frame = path, cv2.resize(im0s.copy(),(1280,720)), getattr(dataset, 'frame', 0)
                    # ---------------------画出区域检测的范围  如果不需要显示，可以在下面注释掉--------------------------
                    # 显示处理后的图像
                    # Detect.show_region_detection(im0)
                    # ---------------------------------------------------------------------------------------------------
                p = Path(p)  # 将路径转换为Path对象
                save_path = str(save_dir / p.name)  # 保存图像的路径
                if len(det):  # 如果检测到目标
                    # 将检测框的坐标从图像尺寸缩放到原图大小
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    CarTotal += len(det)
                    # 打印每类目标的数量
                    # for c in det[:, 5].unique():
                    #     n = (det[:, 5] == c).sum()  # 计算每个类别的数量
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # xywh_bboxs = []     # 初始化列表用于存储检测框和置信度
                    # confs = []
                    # clss = []

                    # # 遍历每个检测框，获取位置信息和类别信息
                    # for *xyxy, conf, cls in reversed(det):
                    #     x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)  # 转换为中心点坐标和宽高
                    #     xywh_obj = [x_c, y_c, bbox_w, bbox_h]  # 存储边界框信息
                    #     xywh_bboxs.append(xywh_obj)  # 添加到列表
                    #     confs.append([conf.item()])  # 存储置信度
                    #     clss.append(int(cls))  # 存储类别

                    # print("输出类别：",clss)
                    # xywhs = torch.Tensor(xywh_bboxs)
                    # confss = torch.Tensor(confs)

                    # print("输出置信度和类别的长度", len(confss), len(clss))

                    # 将检测结果传递给DeepSort进行跟踪

                    # outputs = deepsort.update(xywhs, confss, im0, clss)  # outputs中每一个子数组中的六个数分别是每一个框的左上角x
                    #
                    dets = reversed(det.cpu())
                    dets = sv.Detections.from_yolov5(dets)
                    dets = bytetrack.update_with_detections(dets)
                    # dets:坐标，类别置信度，类别id，识别框id
                    deeosortNum += 1
                    # labels=[]
                    # #测车速
                    # points_ = dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    # points = view_transformer.transform_points(points=points_).astype(int)
                    # for tracker_id, [_, y] in zip(dets.tracker_id, points):
                    #     if 70>y>0:
                    #         coordinates[tracker_id].append(y)

                    # for tracker_id in dets.tracker_id:
                    #     if len(coordinates[tracker_id]) < FPS/2:
                    #         # np.append(labels[tracker_id],0)
                    #         labels.append(f"#{tracker_id}")
                    #     else:
                    #         coordinate_start = coordinates[tracker_id][-1]
                    #         coordinate_end = coordinates[tracker_id][0]
                    #         distance = abs(coordinate_start - coordinate_end)
                    #         time1 = len(coordinates[tracker_id]) / FPS
                    #         speed = distance / time1 * 3.6
                    #         # np.append(labels[tracker_id],int(speed))
                    #         labels.append(f"{tracker_id}:{int(speed)}km/h")

                    # y和右下角xy坐标， 标签， 框序号（追踪索引号）
                    # draw boxes for visualization  绘制用于可视化的边界框
                    # DataNum=DataNum+1
                    # 如果有检测结果，统计车流量并估计速度
                    if len(dets) > 0:
                        # 统计车流量并估计速度
                        # crossed_in, crossed_out = line_zone.trigger(dets)
                        # total_count, down_detail, up_detail, valid_car_info = tool_vehicle.counting(dets, names)  # 统计车流量
                        im0, valid_car_info = tool_vehicle.countingv2(dets, names, view_transformer, max_y=TARGET_HEIGHT, is_draw=False, im0=im0)
                        # 如果有有效的车辆信息， 发送车流量计数及测速结果
                        if len(valid_car_info) > 0:
                            upload_dic = uploadBaseDic
                            # 如果是回放模式，计算历史时间
                            if playback:
                                # time_consume = (int)(frame_idx * (1/25))   #用于计算回放视频的历史当前时间，单位：秒
                                time_consume = (int)(timestamp / 1000)  # 时间戳转化为秒
                                history_time = datetime.fromtimestamp(time_consume + ref_time)  # 计算历史时间
                                upload_dic['detectionTime'] = history_time.strftime('%Y-%m-%d %H:%M:%S')  # 格式化时间
                            else:
                                upload_dic['detectionTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间

                            # 遍历每个车辆，上传图片和车辆信息
                            for pass_car in valid_car_info:
                                cropImg = im0[pass_car[1]:pass_car[3], pass_car[0]:pass_car[2]]  # 裁剪出车辆的图像区域
                                __, buffer = cv2.imencode('.jpg', cropImg)  # 将裁剪图像编码为JPEG格式
                                base64_cropImg = base64.b64encode(buffer.tobytes()).decode('utf-8')  # 编码为Base64格式
                                upload_dic['image'] = base64_cropImg  # 设置上传的图像
                                upload_dic['vehicleModel'] = uploadNameMap[names[int(pass_car[4])]]  # 设置车辆类型
                                upload_dic['speed'] = pass_car[5]  # 设置车速
                                upload_dic['lanesNumber'] = pass_car[6]  # 设置车道号

                                # 构建文件路径和文件名
                                now_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                date_time = datetime.now().strftime('%Y-%m-%d')

                                # 创建一个保存图片目录，以当前日期命名                                     # todo
                                date_path_img = os.path.join(f'{current_save_dir}/{date_time}/img')
                                if not os.path.exists(date_path_img):
                                    os.makedirs(date_path_img)

                                if old_time == date_time:
                                    DataNum = DataNum + 1
                                else:
                                    old_time = date_time
                                    DataNum = 0

                                date_path_txt = os.path.join(f'{current_save_dir}/{date_time}/txt')
                                if not os.path.exists(date_path_txt):
                                    os.makedirs(date_path_txt)

                                now_time = str(now_time) + '_' + str(DataNum) + "_" + str(randint(0, 100))

                                # 保存图片                                                            # todo
                                img_name = f'{now_time}_img.jpg'
                                img_path = os.path.join(date_path_img, img_name)

                                txt_name = f'{now_time}_info.txt'
                                txt_path = os.path.join(date_path_txt, txt_name)

                                TaskWrite = threading.Thread(name='WriteData', target=WriteImgAndTxt, kwargs={"Im": im0, "img_path": img_path, "txt_path": txt_path, "upload_dic": upload_dic, "txt_name": txt_name, "current_folder": date_path_txt})  # 写文件到文件夹
                                TaskWrite.start()

                        # 在UI中画出检测框及估计速度值

                        # tool_vehicle.draw_boxes_speed(im0, outputs, names)
                        # for c in valid_car_info:
                        #     l=f"#{names[int(c[4])]}:{int(c[5])}km/h"
                        #     t_size = cv2.getTextSize(l, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                        #     x1=int(c[0])
                        #     y1=int(c[1])
                        #     x2=int(c[2])
                        #     y2=int(c[3])

                        #     cv2.rectangle(im0, (x1, y1), (x2, y2), [0,0, 255], 3)  # 画出车型的预测框
                        #     cv2.line(im0,(812, 189),(1075, 188),[0,0,0],3)
                        #     cv2.line(im0,(1075, 188),(968, 684),[0,0,0],3)
                        #     cv2.line(im0,(968, 684),(43, 676),[0,0,0],3)
                        #     cv2.line(im0,(43, 676),(812, 189),[0,0,0],3)
                        #     cv2.circle(im0, (x2, y2), radius=4, color=(0, 0, 255), thickness=5)  # 将预测框右下角标出来
                        #     cv2.rectangle(im0, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), [0,0, 255], -1)  # 画出标签的背景框
                        #     cv2.putText(im0, l, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # 写出标签
                # if True:

                # video_savev2(vid_cap=vid_cap,im0=im0)
                # TaskVideo = threading.Thread(name='Video',target=video_savev2, kwargs={"im0":im0,"vid_cap":vid_cap,"save_path":save_path})  # 写文件到文件夹
                # TaskVideo.start()
                # ,"vid_cap":vid_cap,"vid_path":vid_path,
                # video_save(im0,vid_cap,vid_path,save_path,vid_writer)

                # 保存带有检测结果的图像或视频
        # 记录推理时间
        # LOGGER.info( f"{'' if len(det) else '(no detections),'}{(dt[0].dt)*1E3:.1f}ms,{(dt[1].dt)*1E3:.1f}ms,{(dt[2].dt)*1E3:.1f}ms,{(dt[3].dt)* 1E3:.1f}ms")  # 单次全部时间
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms") # Inference Only

    # Print results

    tsum = tuple(x.t * 1E3 for x in dt)  # total speeds
    # DSTime=tsum[3]/deeosortNum
    DSTime = tsum[3] / deeosortNum if deeosortNum != 0 else 0
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'在总共{seen}帧图像中检测到了{CarTotal}辆车,跟踪处理了{deeosortNum}帧图像并识别到了{DataNum}辆有效车辆')
    LOGGER.info(f'Speed: {t[0]:.1f} ms pre-process, {t[1]:.1f}ms inference, {t[2]:.1f}ms NMS, {DSTime:.1f} ms DeepSort')
    # LOGGER.info(f'有效追踪平均处理时间： (%.3fms)/img,{(1, 3, *imgsz)}' % (deepsortTime))
    LOGGER.info('Done. (%.3fs)' % (time.time() - t0))
    cv2.destroyAllWindows()  # 关闭窗口
    if save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # 如果选择更新模型，去除优化器以减少模型大小
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    # run() 结束处添加
    stop_run_flag = True
    cv2.destroyAllWindows()
    LOGGER.info("📽️ run() 自然结束，设置 stop_run_flag = True 以触发重启")
    exit_program()


# 解析命令行参数
def parse_opt():
    parser = argparse.ArgumentParser()
    # 获取当前时间对应的配置
    current_source, current_weights, current_save_dir = get_current_config()
    parser.add_argument('--weights', nargs='+', type=str, default=current_weights,
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=current_source,
                        help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/best1.engine',
    #                     help='model path or triton URL')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt1.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='rtsp://admin:admin12345@192.168.10.3', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='inference/images/test02.avi', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')  # 目标置信度目标筛选
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', type=bool, default=True, help='display tracking video results')  # 显示视频
    parser.add_argument('--view-img', action='store_true', help='show results')  # 不显示视频
    # parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')  # 不保存计数txt文件
    # parser.add_argument('--save-txt', default='True', help='save results to *.txt')  # 保存计数txt文件
    # parser.add_argument('--save', action='store_true', help='do not save images/videos')  # 不保存识别后的图片或视频
    parser.add_argument('--save', type=bool, default=True, help='do not save images/videos')  # 保存识别后的图片或视频
    parser.add_argument('--ref-time', type=str, default='2023-01-01 00:00:00', help='%Y-%m-%d %H:%M:%S')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=current_save_dir, help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")  # deepsort参数设置
    parser.add_argument('--camera-device', type=str, default=device_code1, help='waiting for front enf given')  # todo
    opt = parser.parse_args()  # 解析参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 如果只有一个尺寸，扩展为两倍
    print_args(vars(opt))  # 打印参数
    return opt


# 执行分析功能
def analysisFunction(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查环境要求，排除tensorboard和thop
    run(**vars(opt))  # 调用run函数执行分析


def start_run_thread(opt):
    global current_run_thread, stop_run_flag
    stop_run_flag = False
    current_run_thread = threading.Thread(target=analysisFunction,name='MainCode', args=(opt,))
    current_run_thread.start()


# HTTP服务接口，等待前端发送开始指令并执行主程序
app = Flask(__name__)

# 定义一个接收POST请求的接口
@app.route('/api/analyze-result', methods=['POST'])
def post_start():
    client_ip = request.remote_addr  # 获取请求端的IP地址
    data = request.json  # 获取POST请求中的数据
    if 'deviceCode' not in data or 'videoName' not in data or 'videoStartTime' not in data:
        return jsonify({'error': '缺少必要的字段'}), 400  # 返回错误信息

    # 在这里处理接收到的数据
    device_code = data['deviceCode']
    video_name = data['videoName']
    video_start_time = data['videoStartTime']

    # 执行业务逻辑
    opt = parse_opt()
    opt.source = 'http://' + client_ip + ':' + CLIENT_PORT + '/' + video_name  # 设置视频源
    opt.ref_time = video_start_time  # 设置参考时间
    opt.camera_device = device_code  # 设置设备编码

    # 开始异步执行线程
    task1 = threading.Thread(target=analysisFunction, args=(opt,))
    task1.start()

    LOGGER.info('..........POST..........')
    return jsonify({'message': 'POST请求成功！'})

def check_device_status(device_code):
    url1 = f"http://222.219.137.122:19030/api/device/{device_code}"  # 动态替换{code}为实际的设备编号
    headers1 = {'Content-Type': 'application/x-www-form-urlencoded'}  # 请求头信息
    try:
        # 发送 GET 请求
        response = requests.get(url1, headers=headers1)

        # 检查 HTTP 状态码
        if response.status_code == 200:
            # 尝试解析 JSON 响应
            try:
                data = response.json()
                print(f"设备在线，数据: {data}")
                return True, data  # 返回状态和数据
            except ValueError:
                print(f"设备在线，时间: {response.text}")
                return True, response.text
        else:
            print(f"请求失败 | 设备: {device_code} | 状态码: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"请求异常 | 设备: {device_code} | 错误: {str(e)}")
        return False, None

def write_time_now(file_path,current_date):
    try:
        with open(file_path, 'a') as file:
            file.write(current_date + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

import subprocess
def ResetTime():
    """
    自动同步时间函数
    """
    # 指定NTP服务器的IP地址
    ntp_server_ip = "114.118.7.161"
    # 构建ntpdate命令
    command = ["ntpdate", ntp_server_ip]
    # 调用ntpdate命令来同步时间
    subprocess.run(command, check=True)
    print(f"系统时间已同步到NTP服务器 {ntp_server_ip}")

def addtime(time_string,AddMinues):
    time_obj = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    # 增加10分钟
    time_obj += timedelta(minutes=AddMinues)
    # 将 datetime 对象转换回时间字符串
    new_time_string = time_obj.strftime("%Y-%m-%d %H:%M:%S")
    return new_time_string

def is_valid_time_format(time_str):
    time_format = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(time_str,time_format)
        return True
    except ValueError:
        return False
def sync_device_time():
    config = load_config()
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{current_time}] 尝试调用 SetTimeFromDevice.py 脚本进行时间同步...")
        result = subprocess.run(
            ["sudo", "-S", "/home/uvi/env/yolov5/bin/python", "/home/uvi/Traffic-Survey-v2.1/SetTimeFromDevice.py"],
            input="uvi123\n",  # ⚠️ 注意替换为你的实际sudo密码
            capture_output=True,
            text=True,
            timeout=30
        )
        print("[SetTimeFromDevice 输出]:")
        print(result.stdout)
        if result.stderr:
            print("[错误]:", result.stderr)
        return True
    except Exception as e:
        print(f"[同步异常]: {e}; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return False

def start_check_device_status():
    device_code2 = device_code1  # 获取设备编号
    last_sync_minute = -1  # 初始化上一次同步的分钟值
    TImeHMSNow=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time.sleep(90)
    while True:
        time.sleep(25)  # 睡眠25秒
        CodeNumber=count_threads_by_name('MainCode')
        if CodeNumber < 3:
            #如果总线程数少于3，那么说明有一个线程没有执行，则代码重启
            print(f'状态信号线程监测到总线程数不足，只有：{CodeNumber}，代码重启')
            myLib.get_PID_and_Kill.kill_termination_PID()
        current_minute = int(datetime.now().strftime('%M'))  # 获取当前分钟数
        # current_hour = datetime.now().strftime('%H')  # 如果需要按小时判断也可以用

        check_device_status(device_code2)  # 检查设备状态
        if current_minute % 2 == 0 and current_minute != last_sync_minute:
            print(f"Resend Failed Data Now!{TImeHMSNow}")
            resend_failed_files(['inference/output/night', 'inference/output', 'inference/output/daytime'])  # 调用重发失败文件的函数
        if current_minute % 20 == 0 and current_minute != last_sync_minute:
            # 更新上次同步分钟
            last_sync_minute = current_minute
            state=sync_device_time()  # 调用时间同步函数
            print(f"sync_device_time的时间同步完成，返回为{state}")

def WriteNowTime():
    file_path = 'dateNow.txt'
    TImeHMSNow=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.now()
    current_minute = now.minute-2#提前2分钟记录时间，防止因关机导致的读写错误
    # 判断是否是时间格式，如果是时间格式，则进行时间更新；如果不是时间格式，则重新再写一行时间；
    if current_minute % 3==0:
        try:
            with open(file_path, 'a') as file:
                write_time_now(file_path, TImeHMSNow)
        except Exception as e:
            print(f"时间写入发生错误：{e}:{TImeHMSNow}")


def check_device_time_status(start_time):
    TImeHMSNow=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 记录当前系统时间，每间隔10分钟记录一次，如果当前系统时间比上次时间更早，则不进行记录
    # 在代码执行过程中，保证时间更新不间断
    file_path = 'dateNow.txt'
        # Y-%m-%d_%H
    now = datetime.now()
    current_minute = now.minute-2#提前2分钟记录时间，防止因关机导致的读写错误
    if current_minute % 5 == 0:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        last_date = RebootCheckTime.read_last_line_of_file(file_path)

        # 判断是否是时间格式，如果是时间格式，则进行时间更新；如果不是时间格式，则重新再写一行时间；
        if not is_valid_time_format(last_date):
            try:
                print(f"记录当前时间：{TImeHMSNow}")
                with open(file_path, 'a') as file:
                    file.write('\n' + current_date + '\n')
            except Exception as e:
                print(f"写入文件时发生错误：{e}；{TImeHMSNow}")

        # 如果是时间格式，则进行时间分析和更新；
        else:
            # 解析时间字符串为 datetime 对象
            time1 = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
            time2 = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
            time_difference = abs((time2 - time1).total_seconds())
            # 开发板重启之后，网络还没恢复，系统时间在去的某个时间，这时候可以不断尝试网络更新
            # 如果网络
            if (time1 < time2)&(time_difference>900):  # 如果current_date早于last_date，则说明current_date出现了问题，应该予以更新时间
                print(f"{current_date} 早于 {last_date}，时间误差在15分钟以上;{TImeHMSNow}")
                # 尝试调用服务器进行数据同步，并重新写时间
                # 10s同步一次，持续同步MaxEpoch，如果时间同步成功，则写入file_path
                # AutoReCheckTime.CheckTimeEpoch(file_path, MaxEpoch=10, ntp_server_ip="114.118.7.163")
                sync_device_time()  # 调用时间同步函数

                    # # 如果同步失败，则可能是断网了，可以通过时钟周期来进行代码校正
                    # new_last_date = read_last_line_of_file(file_path)
                    # if new_last_date == last_date:  # 数据没更新，则证明是失败的，则可以last_date记录时间+时钟周期最后更新时间
                    #     end_time = time.perf_counter()  # 获取当前时钟周期
                    #     elapsed_time = end_time - start_time
                    #     AddMinues = elapsed_time / 60
                    #     new_time_string = addtime(current_date, AddMinues)
                    #     write_time_now(file_path, new_time_string)
                    #
                    # start_time = time.perf_counter()  # 更新时钟周期

            elif time1 > time2:  # 如果current_date晚于last_date，说明时间大概率是正常的
                # 计算时间差
                time_difference = abs((time2 - time1).total_seconds())
                if time_difference < 1800:
                    # 如果时间差在30分钟内，则证明时间是连续的，可以记录数据
                    write_time_now(file_path, current_date)
                    start_time = time.perf_counter()  # 获取当前时钟周期
                    print(f"current_date晚于last_date，并在半小时以内;{TImeHMSNow}")
                else:  # 否则先同步数据
                        # AutoReCheckTime.CheckTimeEpoch(file_path, MaxEpoch=10, ntp_server_ip="114.118.7.161")
                        # 确实存在时间同步出现错误，但数据量比较少，可以通过时间同步进行修正
                        # 这里换个ip同步时间，如果同步时间失败，则这些数据可以舍弃
                    new_last_date = RebootCheckTime.read_last_line_of_file(file_path)  # 检查时间同步是否成功，如果成功则更新时钟周期
                    if new_last_date != last_date:  # 数据更新了，则重新获取当前时钟周期
                        start_time = time.perf_counter()
                        print(f"current_date晚于last_date，时间误差在半小时以上;{TImeHMSNow}")
            else:
                print(f"current_date：{current_date} ； last_date：{last_date};TImeHMSNow：{TImeHMSNow}")
                write_time_now(file_path, current_date)


def read_log(log_file_path):
    # 定义函数读取日志文件，返回失败的文件列表
    failed_files = []  # 初始化失败文件列表
    if os.path.exists(log_file_path):  # 检查日志文件是否存在
        with open(log_file_path, 'r') as log_file:  # 打开日志文件
            failed_files = log_file.read().splitlines()  # 按行读取失败文件
    return failed_files  # 返回失败文件列表

def update_device_code(file_content):
    # 定义函数更新设备编码
    device_code_new = device_code1  # 设置新的设备编码
    try:
        data = json.loads(file_content)  # 尝试将文件内容解析为JSON格式
        data["deviceCode"] = device_code_new  # 更新设备编码
        return json.dumps(data)  # 返回更新后的JSON字符串
    except json.JSONDecodeError:
        print("无法解析 JSON 格式，返回原始内容。")  # 解析失败时返回原始内容
        return file_content

def send_file_to_server(file_path):
    # 定义函数发送文件到服务器
    headers = {'Content-Type': 'application/json'}  # 设置请求头为JSON格式
    try:
        with open(file_path, 'r') as f:  # 打开指定路径的文件
            file_content = f.read()  # 读取文件内容
        # 更新设备编码
        updated_content = update_device_code(file_content)  # 更新设备编码
        # 尝试将 updated_content 解析为 JSON 格式
        try:
            upload_dic = json.loads(updated_content)  # 尝试将更新后的内容解析为JSON
        except json.JSONDecodeError:
            print(f"无法解析文件 {os.path.basename(file_path)} 的 JSON 格式。将发送原始内容。")
            upload_dic = {"file_name": os.path.basename(file_path), "file_content": updated_content}  # 发送原始内容
        print(f"正在发送文件: {os.path.basename(file_path)}")  # 打印发送的文件路径
        r_json = requests.post(upload_addr3, headers=headers, data=json.dumps(upload_dic))  # 发送POST请求
        # 检查是否发送成功
        if r_json.status_code == 200:  # 如果响应状态码为200，表示成功
            print(f"文件发送成功√√√√")
            return True  # 返回成功状态
        else:  # 否则打印失败信息
            print(f"文件发送失败，状态码: {r_json.status_code}××××")
            print(f"服务器响应: {r_json.text}")  # 打印服务器响应内容
            print(f"文件{file_path}文件传输失败是非网络问题导致，后续不再重传")  # 打印服务器响应内容

            return True  # 返回失败状态，除了网络问题之外，其它发送失败任务均不再进行处理
    except RequestException as e:  # 捕获网络请求异常
        print(f"发送文件时网络请求异常。。。")  # 打印错误信息
        # print(f"发送文件时网络发生错误: {e}××××")  # 打印错误信息
        return False  # 返回失败状态

# 更新日志文件（移除成功发送的文件）
def update_log(log_file_path, failed_files):
    with open(log_file_path, 'w') as log_file:  # 打开日志文件进行写入
        for file in failed_files:
            log_file.write(f"{file}\n")  # 将失败的文件逐行写入日志文件

def resubmitlTxtData():
    # 定义函数定期重发文本数据
    while True:
        resend_failed_files(['inference/output/night', 'inference/output', 'inference/output/daytime'])  # 调用重发失败文件的函数
        time.sleep(60)  # 每隔60秒调用一次

def remove_line(fileName,lineToSkip):
    with open(fileName,'r', encoding='utf-8') as read_file:
        lines = read_file.readlines()
    currentLine = 1
    with open(fileName,'w', encoding='utf-8') as write_file:
        for line in lines:
            if currentLine == lineToSkip:
                pass
            else:
                write_file.write(line)
            currentLine += 1

def resend_failed_files(possible_save_dirs):
    error_dir = "inference/error_logger"
    Num=0
    for log_file_name in os.listdir(error_dir):
        if not log_file_name.endswith('.txt'):
            continue

        date_str = log_file_name[:10]  # e.g. 2024-10-19
        log_file_path = os.path.join(error_dir, log_file_name)
        failed_files = read_log(log_file_path)
        line_i = 0
        if len(failed_files)>0:
            print(f"共有{len(failed_files)}辆车需要数据重传，请等待数据重传完成...")
        else:
            continue

        for file_name in failed_files:
            line_i += 1
            file_found = False
            

            # 依次尝试每个保存路径
            for base_dir in possible_save_dirs:
                candidate_path = os.path.join(base_dir, date_str, "txt", file_name)
                if os.path.exists(candidate_path):
                    print(f"✅ 找到文件: {file_name}，正在重传...")
                    if send_file_to_server(candidate_path):
                        print("重传成功")
                        Num = Num + 1
                        remove_line(log_file_path, line_i)
                        line_i -= 1
                    else:
                        print(f"重传失败: {file_name}")
                    file_found = True
                    break  # 找到即终止查找

            if not file_found:
                print(f"文件{file_name}在以下目录中都未找到:")
                remove_line(log_file_path, line_i)
                line_i -= 1
                for base_dir in possible_save_dirs:
                    print(f"  - {os.path.join(base_dir, date_str, 'txt')}")
    if Num>0:
        print(f"共{Num}辆车重发完成")

def config_checker(need_restart = False):
    global current_source, current_weights, current_save_dir, model, dataset
    global stop_run_flag, current_run_thread
    start_time = time.perf_counter() #开始记录时间的时钟周期
    time.sleep(90)
    while True:
        WriteNowTime()
        CodeNumber=count_threads_by_name('MainCode')
        if CodeNumber < 3:
            #如果总线程数少于3，那么说明有一个线程没有执行，则代码重启
            print(f'视频流线程监测到总线程数不足，只有：{CodeNumber}，代码重启')
            myLib.get_PID_and_Kill.kill_termination_PID()
        
        now = datetime.now()
        current_minute = now.minute-2#提前3分钟记录时间，防止因关机导致的读写错误
        # 判断是否是时间格式，如果是时间格式，则进行时间更新；如果不是时间格式，则重新再写一行时间；
        if current_minute % 30==0:
            sync_device_time() #30分钟自动同步时钟一次
        
        check_device_time_status(start_time)
        new_source, new_weights, new_save_dir = get_current_config()
        
        # 1. 配置变了
        if new_source != current_source or new_weights != current_weights or new_save_dir != current_save_dir:
            print(f"🌀 检测到配置变化，从 {current_source} 切换到 {new_source}")
            print(f"模型从 {current_weights} 切换到 {new_weights}")
            print(f"输出目录切换到 {new_save_dir}")
            print("🌀 检测到配置变化，准备重启")
            need_restart = True

        # 2. run() 自然结束或超时退出
        restart_count = 0
        if stop_run_flag and (not current_run_thread or not current_run_thread.is_alive()):
            print("🔄 run() 已结束，准备重启")
            restart_count += 1
            if restart_count > 2:
                print("❌ run() 重启超过3次仍失败，程序退出")
            need_restart = True

        if need_restart:
            if current_run_thread and current_run_thread.is_alive():
                print("⏳ 发送退出信号给旧线程...")
                stop_run_flag = True  # ① 通知退出
                myLib.get_PID_and_Kill.kill_termination_PID()

                '''
                # 旧代码，推出线程信号，代码存在的问题：只推出了当前线程，未正式推出程序，线程重新启动不会重新加载视频流
                print("⏳ 发送退出信号给旧线程...")
                stop_run_flag = True  # ① 通知退出
                os._exit(0)  # 强制退出所有线程，立即关闭
                current_run_thread.join()  # ② 等待退出
                print("✅ 旧线程已退出")
                '''

            stop_run_flag = False  # ③ 重置退出标志，准备启动新线程

            # 更新配置
            opt.source = new_source
            opt.weights = new_weights
            opt.project = new_save_dir
            # 使用现有 opt 重启（不需要变更 source/weights）
            start_run_thread(opt)

            # 记录最新配置
            current_source, current_weights, current_save_dir = new_source, new_weights, new_save_dir
            print("🚀 线程重启完成")

        time.sleep(50)


def clean_file_keep_last_lines(file_path, keep_lines=100):
    try:
        if not os.path.exists(file_path):
            print(f"{os.path.basename(file_path)} 不存在，无需清理。")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) > keep_lines:
            lines = lines[-keep_lines:]  # 只保留最后 keep_lines 行

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"✅ {os.path.basename(file_path)} 清理完成，只保留最后 {keep_lines} 行。")
        else:
            print(f"✅ {os.path.basename(file_path)} 行数少于 {keep_lines}，无需清理。")
    except Exception as e:
        print(f"⚠️ 清理 {os.path.basename(file_path)} 出错: {e}")

def count_threads_by_name(target_name='MainCode'):
    count = 0
    for thread in threading.enumerate():
        if thread.name == target_name:
            count += 1
    return count

def delete_errorlog():
    """
    删除 error_dir 中为空的错误日志文件。
    """
    error_dir = "inference/error_logger"
    deleted_count = 0
    if not os.path.exists(error_dir):
        print("❗ 错误日志目录不存在，无需删除空日志。")
        return

    for file_name in os.listdir(error_dir):
        file_path = os.path.join(error_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    print(f"🗑️ 删除空日志文件: {file_name}")
                    deleted_count += 1
            except Exception as e:
                print(f"⚠️ 无法删除 {file_name}: {e}")

    print(f"✅ 空日志清理完成，共删除 {deleted_count} 个文件。")


if __name__ == '__main__':

    global opt
    opt = parse_opt()

    # 只在程序启动时清理一次 dateNow.txt
    file_path = 'dateNow.txt'
    clean_file_keep_last_lines(file_path, keep_lines=300)
    #  程序启动之前，先校正时间
    last_date = RebootCheckTime.read_last_line_of_file(file_path)
    print(f"读取文档{file_path}中的时间为: {last_date}")
    adjusted_datetime_str = RebootCheckTime.adjust_datetime_string1(last_date,addBiasTime=1)
    print(f"调整时间为：{adjusted_datetime_str}")
    if RebootCheckTime.adjust_Time(adjusted_datetime_str): # 先手动调整时间为adjusted_datetime_str
         print("手动校正时间完成！")
    
    print("代码重启后，先尝试重传传输失败的车辆")
    resend_failed_files(['inference/output/night', 'inference/output', 'inference/output/daytime'])
    print("代码重启后，先删除空的error文件")
    delete_errorlog()

    # 删除文件夹
    base_directory = '/home/uvi/Traffic-Survey-v2.1/inference/output'
    RebootCheckTime.delete_old_folders(base_directory)

    if sync_device_time(): #后自动同步
        print("自动校正时间完成")


    # 初始化全局变量
    current_source, current_weights, current_save_dir = get_current_config()
    start_run_thread(opt)

    # start_check_device_status()
    task2 = threading.Thread(target=start_check_device_status,name='MainCode', args=())  # 创建线程检查设备状态
    task2.start()  # 启动线程

    # 启动配置检查线程
    checker_thread = threading.Thread(target=config_checker,name='MainCode', daemon=True)
    checker_thread.start()

