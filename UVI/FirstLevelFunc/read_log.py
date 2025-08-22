
import os
def read_log(log_file_path):
    # 定义函数读取日志文件，返回失败的文件列表
    failed_files = []  # 初始化失败文件列表
    if os.path.exists(log_file_path):  # 检查日志文件是否存在
        with open(log_file_path, 'r') as log_file:  # 打开日志文件
            failed_files = log_file.read().splitlines()  # 按行读取失败文件
    return failed_files  # 返回失败文件列表
