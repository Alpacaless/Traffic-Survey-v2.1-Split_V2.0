import argparse
import UVI.SecondLevelFunc.get_current_config as get_current_config
from yolov5.utils.general import print_args
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)

print("result",result)

# 解析命令行参数
def parse_opt():
    parser = argparse.ArgumentParser()
    
    # 读取参数
    

    current_data = result['current_data']
    current_imgsz = result['current_imgsz']
    current_conf_thres = result['current_conf_thres']
    current_iou_thres = result['current_iou_thres']
    current_max_det = result['current_max_det']
    current_device = result['current_device']
    current_view_img = result['current_view_img']
    current_save_txt = result['current_save_txt']
    current_save = result['current_save']
    current_ref_time = result['current_ref_time']
    current_classes = result['current_classes']
    current_agnostic_nms = result['current_agnostic_nms']
    current_augment = result['current_augment']
    current_visualize = result['current_visualize']
    current_update = result['current_update']
    current_project_dir = result['current_project_dir']
    current_name = result['current_name']
    current_exist_ok = result['current_exist_ok']
    current_half = result['current_half']
    current_dnn = result['current_dnn']
    current_vid_stride = result['current_vid_stride']
    current_camera_device = result['current_camera_device']
    device_code1 = result['device_code1']   
    
    if device_code1 != "0021145319062069":   # luxi的device_code
        current_weights = result['current_weights']
        current_source = result['current_source']
        current_project_dir = result['current_project_dir']
    elif device_code1 == "0021145319062069":
        current_weights, current_source, current_project_dir = get_current_config.get_current_config()
 
    parser.add_argument('--weights', nargs='+', type=str, default=current_weights,help='model path or triton URL')
    parser.add_argument('--source', type=str, default=current_source,help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=current_data, help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=current_imgsz, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=current_conf_thres, help='confidence threshold')  # 目标置信度目标筛选
    parser.add_argument('--iou-thres', type=float, default=current_iou_thres, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=current_max_det, help='maximum detections per image')
    parser.add_argument('--device', default=current_device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # parser.add_argument('--view-img', type=bool, default=True, help='display tracking video results')  # 显示视频
    parser.add_argument('--view-img', action='store_true', default=current_view_img, help='show results')  # 不显示视频

    # parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')  # 不保存计数txt文件
    parser.add_argument('--save-txt', default=current_save_txt, help='save results to *.txt')  # 保存计数txt文件

    # parser.add_argument('--save', action='store_true', help='do not save images/videos')  # 不保存识别后的图片或视频
    parser.add_argument('--save', type=bool, default=current_save, help='do not save images/videos')  # 保存识别后的图片或视频

    parser.add_argument('--ref-time', type=str, default=current_ref_time, help='%Y-%m-%d %H:%M:%S')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=current_agnostic_nms, help='class-agnostic NMS')
    parser.add_argument('--augment', default=current_augment, help='augmented inference')
    parser.add_argument('--visualize', default=current_visualize, help='visualize features')
    parser.add_argument('--update', default=current_update, help='update all models')
    parser.add_argument('--project', default=current_project_dir, help='save results to project/name')
    parser.add_argument('--name', default=current_name, help='save results to project/name')
    parser.add_argument('--exist-ok', default=current_exist_ok, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=current_vid_stride, help='video frame-rate stride')
    parser.add_argument('--camera-device', type=str, default=device_code1, help='waiting for front enf given')  # todo
    
    
    opt = parser.parse_args()  # 解析参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 如果只有一个尺寸，扩展为两倍
    print_args(vars(opt))  # 打印参数
    return opt