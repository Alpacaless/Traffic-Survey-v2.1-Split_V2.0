
import os
import json
import requests
from requests.exceptions import RequestException
import UVI.FirstLevelFunc.update_device_code as update_device_code


# 数据上传地址
upload_addr3 = 'http://222.219.137.122:19030/api/analyze-result'  # 车辆信息上传地址

def send_file_to_server(file_path, device_code1):
    # 定义函数发送文件到服务器
    headers = {'Content-Type': 'application/json'}  # 设置请求头为JSON格式
    try:
        with open(file_path, 'r') as f:  # 打开指定路径的文件
            file_content = f.read()  # 读取文件内容

        # 更新设备编码
        updated_content = update_device_code.update_device_code(file_content, device_code1)  # 更新设备编码

        # 尝试将 updated_content 解析为 JSON 格式
        try:
            upload_dic = json.loads(updated_content)  # 尝试将更新后的内容解析为JSON
        except json.JSONDecodeError:
            print(f"无法解析文件 {os.path.basename(file_path)} 的 JSON 格式。将发送原始内容。")
            upload_dic = {"file_name": os.path.basename(file_path), "file_content": updated_content}  # 发送原始内容
        print(f"正在发送文件: {os.path.basename(file_path)}")  # 打印发送的文件路径
        r_json = requests.post(upload_addr3, headers=headers, data=json.dumps(upload_dic))  # 发送POST请求
        # 检查是否发送成功
        if r_json.status_code == 200:  # 如果响应状态码为200，表示成功
            print(f"文件发送成功√√√√")
            return True  # 返回成功状态
        else:  # 否则打印失败信息
            print(f"文件发送失败，状态码: {r_json.status_code}××××")
            print(f"服务器响应: {r_json.text}")  # 打印服务器响应内容
            print(f"文件{file_path}文件传输失败是非网络问题导致，后续不再重传")  # 打印服务器响应内容

            return True  # 返回失败状态，除了网络问题之外，其它发送失败任务均不再进行处理
    except RequestException as e:  # 捕获网络请求异常
        print(f"发送文件时网络请求异常。。。")  # 打印错误信息
        # print(f"发送文件时网络发生错误: {e}××××")  # 打印错误信息
        return False  # 返回失败状态
