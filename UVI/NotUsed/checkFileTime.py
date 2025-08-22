import os
import time

import os
import time

def find_latest_modified_folder_by_content(root_dir):
    latest_folder = None
    latest_content_mod_time = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_latest_mod_time = 0
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_mod_time = os.path.getmtime(file_path)
            if file_mod_time > folder_latest_mod_time:
                folder_latest_mod_time = file_mod_time
        
        if folder_latest_mod_time > latest_content_mod_time:
            latest_content_mod_time = folder_latest_mod_time
            latest_folder = dirpath  # 这里使用 dirpath，因为它是文件夹的完整路径（不包括文件夹名）

    if latest_folder:
        latest_folder_name = os.path.basename(latest_folder)
        return latest_folder_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_content_mod_time))
    else:
        return None, None

# 使用示例（同上）


# 指定文件路径
file_path = 'dateNow.txt'

# 获取文件的最后修改时间
modification_time = os.path.getmtime(file_path)

# 将时间戳转换为可读格式
readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modification_time))

print(f"The last modification time of {file_path} is: {readable_time}")