import os
import json
import cv2
from datetime import datetime

import UVI.FirstLevelFunc.is_within_time_range as is_within_time_range
import UVI.SecondLevelFunc.reupload_from_last_10_percent as reupload_from_last_10_percent
import UVI.SecondLevelFunc.resend_midsizecar as resend_midsizecar
import UVI.SecondLevelFunc.send_file_to_server as send_file_to_server
import UVI.FirstLevelFunc.log_error as log_error
error_dir = "inference/error_logger"  # 错误日志文件保存目录
os.makedirs(error_dir, exist_ok=True)  # 创建错误日志目录

# 全局变量：已上传文件的路径存储
uploaded_files = []
upload_count = 0  # 文件上传计数器

last_save_time = None


def WriteImgAndTxt(Im,img_path,txt_path,upload_dic,txt_name, current_folder, device_code1):
    global upload_count, uploaded_files, last_save_time

    date_time_h = datetime.now().strftime('%Y-%m-%d_%H')
    error_name = f'{date_time_h}_error_logg.txt'
    error_path = os.path.join(error_dir, error_name)
    # cv2.imwrite(img_path, Im)  # 保存图片

    # 检查上次保存的时间和当前时间的间隔
    current_time = datetime.now()
    if is_within_time_range.is_within_time_range():
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
    if send_file_to_server.send_file_to_server(txt_path, device_code1):
        now_time = datetime.now()
        uploaded_files.append((txt_path, now_time))  # 记录上传文件路径及其时间戳
        upload_count += 1

        # # 每上传20个文件，从当前文件夹中的后10%文件中随机选择一个重新上传
        # if  is_within_time_range.is_within_time_range():
        #     if upload_count % 18 == 0:
        #         reupload_from_last_10_percent.reupload_from_last_10_percent(current_folder,device_code1)
        #     if upload_count % 50 == 0:
        #         resend_midsizecar.resend_midsizecar(current_folder)
    else:
        log_error.log_error(txt_name)

    return 0
