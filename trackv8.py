import sys
sys.path.insert(0, './yolov5')
import argparse,platform
import threading,pathlib
from flask import jsonify

import UVI.ThirdLevelFunc.parse_opt as parse_opt
import UVI.ThirdLevelFunc.clean_file_keep_last_lines as clean_file_keep_last_lines
import UVI.FirstLevelFunc.read_last_line_of_file as read_last_line_of_file
import UVI.ThirdLevelFunc.adjust_datetime_string1 as adjust_datetime_string1
import UVI.ThirdLevelFunc.adjust_time as adjust_time
import UVI.ThirdLevelFunc.resend_failed_files as resend_failed_files
import UVI.ThirdLevelFunc.delete_errorlog as delete_errorlog
import UVI.ThirdLevelFunc.delete_old_folders as delete_old_folders
import UVI.ThirdLevelFunc.sync_device_time as sync_device_time
import UVI.ThirdLevelFunc.start_run_thread as start_run_thread
import UVI.ThirdLevelFunc.start_check_device_status as start_check_device_status
import UVI.ThirdLevelFunc.config_checker as config_checker




plt = platform.system()
if plt != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath



if __name__ == '__main__':

    global opt
    opt = parse_opt.parse_opt()
    print("opt",opt)

    # 只在程序启动时清理一次 dateNow.txt
    file_path = 'dateNow.txt'
    clean_file_keep_last_lines.clean_file_keep_last_lines(file_path, keep_lines=300)

    #  程序启动之前，先校正时间     
    last_date = read_last_line_of_file.read_last_line_of_file(file_path)
    print(f"读取文档{file_path}中的时间为: {last_date}")
   
    adjusted_datetime_str = adjust_datetime_string1.adjust_datetime_string1(last_date,addBiasTime=1)

    print(f"调整时间为：{adjusted_datetime_str}")
    if adjust_time.adjust_Time(adjusted_datetime_str): # 先手动调整时间为adjusted_datetime_str
         print("手动校正时间完成！")
    #------------------------------
    print("代码重启后，先尝试重传传输失败的车辆")
    resend_failed_files.resend_failed_files(['inference/output/night', 'inference/output', 'inference/output/daytime'])
    print("代码重启后，先删除空的error文件")
    delete_errorlog.delete_errorlog()

    # 删除文件夹
    base_directory = r'inference\output'
    delete_old_folders.delete_old_folders(base_directory)

    if sync_device_time.sync_device_time(): #后自动同步
        print("自动校正时间完成")

    ### run
    start_run_thread.start_run_thread(opt)


    # start_check_device_status()
    task2 = threading.Thread(target=start_check_device_status.start_check_device_status,name='MainCode', args=())  # 创建线程检查设备状态
    task2.start()  # 启动线程

    # 启动配置检查线程
    checker_thread = threading.Thread(target=config_checker.config_checker,name='MainCode', daemon=True)
    checker_thread.start()

