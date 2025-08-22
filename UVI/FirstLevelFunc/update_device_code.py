
import json

def update_device_code(file_content, device_code1):
    # 定义函数更新设备编码
    device_code_new = device_code1  # 设置新的设备编码
    try:
        data = json.loads(file_content)  # 尝试将文件内容解析为JSON格式
        data["deviceCode"] = device_code_new  # 更新设备编码
        return json.dumps(data)  # 返回更新后的JSON字符串
    except json.JSONDecodeError:
        print("无法解析 JSON 格式，返回原始内容。")  # 解析失败时返回原始内容
        return file_content
