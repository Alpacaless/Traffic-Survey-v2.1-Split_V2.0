import requests

def check_device_status(device_code):
    url1 = f"http://222.219.137.122:19030/api/device/{device_code}"  # 动态替换{code}为实际的设备编号
    headers1 = {'Content-Type': 'application/x-www-form-urlencoded'}  # 请求头信息
    try:
        # 发送 GET 请求
        response = requests.get(url1, headers=headers1)

        # 检查 HTTP 状态码
        if response.status_code == 200:
            # 尝试解析 JSON 响应
            try:
                data = response.json()
                print(f"设备在线，数据: {data}")
                return True, data  # 返回状态和数据
            except ValueError:
                print(f"设备在线，时间: {response.text}")
                return True, response.text
        else:
            print(f"请求失败 | 设备: {device_code} | 状态码: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"请求异常 | 设备: {device_code} | 错误: {str(e)}")
        return False, None
