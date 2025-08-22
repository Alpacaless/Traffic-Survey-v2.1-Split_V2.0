import os
import random
import json
import UVI.SecondLevelFunc.update_and_reupload as update_and_reupload

def resend_midsizecar(current_folder):
    try:
        # 获取当前文件夹中的所有文件并按修改时间排序
        all_files = [
            os.path.join(current_folder, f) for f in os.listdir(current_folder)
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

        # 筛选midsizecar和smalltruck的文件
        filtered_files = []
        for file_path in candidates:
            try:
                with open(file_path, 'r') as file:
                    content = json.load(file)
                    vehicle_model = content.get('vehicleModel', '').lower()
                    if vehicle_model in ['midsizecar', 'smalltruck']:
                        filtered_files.append(file_path)
            except Exception as e:
                print(f"无法处理文件 {file_path}: {e}")

        if not filtered_files:
            print("未找到符合条件的文件。")
            return

        # 随机选择一个文件上传
        random_file = random.choice(filtered_files)
        update_and_reupload.update_and_reupload(random_file)
    except Exception as e:
        print(f"重新上传时发生错误: {e}")
