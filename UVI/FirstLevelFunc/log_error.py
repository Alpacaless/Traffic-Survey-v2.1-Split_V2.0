import os
from datetime import datetime

error_dir = "inference/error_logger"  # 错误日志文件保存目录
os.makedirs(error_dir, exist_ok=True)  # 创建错误日志目录

def log_error(txt_name):
    date_time_h = datetime.now().strftime('%Y-%m-%d_%H')
    error_name = f'{date_time_h}_error_logg.txt'
    error_path = os.path.join(error_dir, error_name)

    with open(error_path, "a") as error_file:
        error_file.write(txt_name + "\n")