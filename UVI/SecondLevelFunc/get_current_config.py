import UVI.FirstLevelFunc.load_config as load_config
import UVI.FirstLevelFunc.parse_time as parse_time
from datetime import datetime


def get_current_config():
    configs = load_config.load_config()
    now = datetime.now().time()

    for config in configs['configurations']:
        start_str, end_str = config['time_range'].split('-')
        start_time = parse_time.parse_time(start_str)
        end_time = parse_time.parse_time(end_str)

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