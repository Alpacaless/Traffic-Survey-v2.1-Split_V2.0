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


# HTTP服务接口，等待前端发送开始指令并执行主程序
app = Flask(__name__)
CLIENT_PORT = '6060'  # 客户端端口号
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

