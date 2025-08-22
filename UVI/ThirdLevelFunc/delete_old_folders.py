import os
from datetime import datetime, date, timedelta

def delete_old_folders(base_path, days_threshold=45):
    # 获取当前日期
    current_date = date.today()
    # 计算阈值日期
    threshold_date = current_date - timedelta(days=days_threshold)
    
    # 遍历 base_path 下的所有文件和文件夹
    for item in os.listdir(base_path):
        # 检查是否为文件夹
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # 尝试将文件夹名转换为日期对象
            try:
                folder_date = datetime.strptime(item, '%Y-%m-%d').date()
            except ValueError:
                # 如果转换失败，则不是日期命名的文件夹，跳过
                continue
            
            # 比较文件夹日期和阈值日期
            item_path1=os.path.join(item_path,'txt/')#删除以txt命名的文件夹
            if folder_date < threshold_date:
                # 删除文件夹及其内容
                print(f"Deleting folder: {item_path1}")
                
                if os.path.exists(item_path1):
                    shutil.rmtree(item_path1)
            else:         # 文件夹日期在阈值之内，不删除
                print(f"Keeping folder: {item_path1}")
        else:
            # 不是文件夹，跳过
            print(f"跳过非文件夹: {item_path}")


