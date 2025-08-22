import os
import random
import UVI.SecondLevelFunc.update_and_reupload as update_and_reupload
def reupload_from_last_10_percent(current_folder,device_code):
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
        update_and_reupload.update_and_reupload(random_file,device_code)
    except Exception as e:
        print(f"重传错误: {e}")
