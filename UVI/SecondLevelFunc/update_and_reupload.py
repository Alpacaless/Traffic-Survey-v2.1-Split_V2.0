import os
import json
import re
from datetime import datetime
import UVI.SecondLevelFunc.send_file_to_server as send_file_to_server

def update_and_reupload(file_path,device_code):
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"重传文件: {file_path}，更新为当前时间：{now_time}")

    # 修改文件中的时间戳
    with open(file_path, 'r') as f:
        content = f.read()

    try:
        # 使用正则提取第一个JSON对象
        match = re.match(r'({.*?})', content, re.DOTALL)
        if not match:
            print(f"文件 {file_path} 不包含有效的JSON对象。")
            return
        # 提取第一个JSON对象
        upload_dic = json.loads(match.group(1))
        # upload_dic = json.loads(content)
        upload_dic['detectionTime'] = now_time  # 更新为当前时间

        with open(file_path, 'w') as f:
            json.dump(upload_dic, f)

        # 重新上传文件
        send_file_to_server.send_file_to_server(file_path,device_code)
    except json.JSONDecodeError as e:
        print(f"文件内容格式错误，无法解析为JSON: {e}")
    except Exception as e:
        print(f"更新文件时发生错误: {e}")
