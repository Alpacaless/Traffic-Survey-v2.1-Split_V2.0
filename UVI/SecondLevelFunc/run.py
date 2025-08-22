import sys
sys.path.insert(0, './yolov5')
# from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


# from yolov5.Detect import Detect
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.downloads import attempt_download
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                  colorstr, increment_path, non_max_suppression, print_args, scale_boxes,
                                  strip_optimizer)
from yolov5.utils.torch_utils import select_device, smart_inference_mode



import matplotlib.pyplot as plt
import supervision as sv
import time, datetime
from pathlib import Path
import cv2,os,torch,threading
import torch.backends.cudnn as cudnn
from random import randint
import requests, json, base64

import numpy as np
from datetime import datetime


import UVI.SecondLevelFunc.countingv2 as countingv2
import UVI.SecondLevelFunc.get_current_config as get_current_config
import UVI.SecondLevelFunc.WriteImgAndTxt as WriteImgAndTxt
import UVI.FirstLevelFunc.load_config as load_config


configs = load_config.load_config()

uploadNameMap = configs['uploadNameMap']
stop_run_flag = configs['stop_run_flag']
FPS = configs['FPS']
UP_LINE = configs['up_line']
DOWN_LINE = configs['down_line']
TARGET_WIDTH = configs['target_width']
TARGET_HEIGHT = configs['target_height']
current_save_dir = configs['current_project_dir']

COUNT_NUMBER = configs['COUNT_NUMBER']

SOURCE = configs['source_points']

# 计算 LINE_start (取 [0] 和 [3] 点的中点)
LINE_start = [(SOURCE[0][0] + SOURCE[3][0]) / 2, 
              (SOURCE[0][1] + SOURCE[3][1]) / 2]

# 计算 LINE_end (取 [1] 和 [2] 点的中点)
LINE_end = [(SOURCE[1][0] + SOURCE[2][0]) / 2, 
            (SOURCE[1][1] + SOURCE[2][1]) / 2]


start, end = sv.Point(x=0, y=int(LINE_start[1])), sv.Point(x=1920, y=int(LINE_end[1]))
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

global valid_car_info
valid_car_info = []

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        # source = source.astype(np.float32)
        # target = target.astype(np.float32)
        # 确保 source 和 target 是 NumPy 数组
        source = np.array(source, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def exit_program():
    print("退出程序，终止线程")
    os._exit(0)  # 立即强制退出整个Python程序（包括所有线程）


# 启用智能推理模式
@smart_inference_mode()
def run(weights=configs['current_weights'],  # model path or triton URL
        source=configs['current_source'],  # file/dir/URL/glob/screen/0(webcam)
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
        camera_device='192.168.10.3',
        ):
    
    lost_frame_count = 0  # 初始化丢帧计数器
    MAX_LOST_FRAMES = 30  # 连续丢帧30次则重启，大约30秒
    global stop_run_flag
    
    
    # 初始化 ByteTrack
    bytetrack= sv.ByteTrack(track_activation_threshold=0.25,lost_track_buffer=FPS,minimum_matching_threshold=0.8,frame_rate=FPS,minimum_consecutive_frames=1)
    
    # 初始化上传的字典，包含设备编码、图像等信息
    uploadBaseDic = {"deviceCode": '28254', "image": '0', "vehicleModel": '0', "speed": 0, "lanesNumber": 0,
                     "detectionTime": '0'}
    uploadBaseDic['deviceCode'] = camera_device

    # 初始化工具类 ToolVehicle

    print("source",source)
    print("weights",weights)
    print("camera_device",camera_device)

    #测速
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    

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

    # 注意：请在循环开始前定义这个文件（只写一次）

    open("track.txt", "w").close()  # 清空文件（只在第一帧）

    # 遍历每一帧图像，执行推理和跟踪
    for frame_idx, (path, im, im0s, vid_cap, timestamp, ret_flag) in enumerate(dataset):


        # print(type(path), type(im), type(im0s))
        #
        # print("预处理后的图像 im shape: ", im[0])
        # print("原始图像 im0s shape: ", im0s[0])
        # print("路径 path: ", path)
        # print("预处理后的图像 vid_cap: ", vid_cap)
        # print("原始图像 timestamp: ", timestamp)
        # print("路径 ret_flag: ", ret_flag)
        # break  # 查看一帧后退出


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

        # img = im.copy()  # 深拷贝
        # img = img.astype(np.float32)
        #
        #
        # im_np = np.transpose(img, (1, 2, 0))  # (C, H, W) → (H, W, C)


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
                    # print(frame_idx)
                    # 将检测框的坐标从图像尺寸缩放到原图大小

                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    CarTotal += len(det)








                    # x1, y1, x2, y2 = map(int, det[0, :4].tolist())
                    # #
                    # # # 2️⃣ 画框（绿色，线宽 2）
                    # cv2.rectangle(im0, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    # #
                    # # # 3️⃣ 显示图像（使用 OpenCV 的窗口）
                    # cv2.imshow("Image with Bounding Box", im0)
                    # cv2.waitKey(0)  # 等待键盘输入
                    # cv2.destroyAllWindows()





                    det_cpu = det.cpu()
                    xyxy = det_cpu[:, :4].numpy()  # 转成numpy ndarray
                    confidence = det_cpu[:, 4].numpy()
                    class_id = det_cpu[:, 5].int().numpy()  # 如果整数类别也转
                    dets = sv.Detections(
                        xyxy = xyxy,
                        confidence = confidence,
                        class_id = class_id
                    )
                    dets = bytetrack.update_with_detections(dets)

                    # dets = reversed(det.cpu())
                    # dets = sv.Detections.from_yolov5(dets)
                    # dets = bytetrack.update_with_detections(dets)
                    # # dets:坐标，类别置信度，类别id，识别框id





                    # 每一帧处理后，保存结果
                    with open("track.txt", "a") as f:
                        for det in dets:  # results 是 bytetrack.update_with_detections(dets) 返回的
                            tid = det[4]
                            x1, y1, x2, y2 = det[0]  # 坐标为 xyxy 格式
                            score = det[2]
                            class_id = det[3]

                            # 转为 tlwh（左上角 + 宽高）
                            x = x1
                            y = y1
                            w = x2 - x1
                            h = y2 - y1

                            # frame_id +1 是为了从 1 开始计数（符合 MOT 格式）
                            line = f"{frame_idx + 1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f},{class_id},-1,-1\n"
                            # line = f"{i + 1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f},-1,-1,-1\n"
                            f.write(line)




                    deeosortNum += 1
                    # 如果有检测结果，统计车流量并估计速度
                    if len(dets) > 0:
                        # 统计车流量并估计速度
                        # crossed_in, crossed_out = line_zone.trigger(dets)
                        # total_count, down_detail, up_detail, valid_car_info = tool_vehicle.counting(dets, names)  # 统计车流量
                        global COUNT_NUMBER
                        im0, valid_car_info, COUNT_NUMBER = countingv2.countingv2(count=COUNT_NUMBER, dets=dets, cls_names=names, view_transformer=view_transformer, FPS=FPS, max_y=TARGET_HEIGHT, up_line=UP_LINE, down_line=DOWN_LINE, is_draw=False, im0=im0)
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

                                TaskWrite = threading.Thread(name='WriteData', target=WriteImgAndTxt.WriteImgAndTxt, kwargs={"Im": im0, "img_path": img_path, "txt_path": txt_path, "upload_dic": upload_dic, "txt_name": txt_name, "current_folder": date_path_txt, "device_code1": configs['device_code1']})  # 写文件到文件夹
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
