import subprocess
from datetime import datetime

def sync_device_time():
    ##################
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{current_time}] 尝试调用 SetTimeFromDevice.py 脚本进行时间同步...")
        result = subprocess.run(
            ["sudo", "-S", "/home/uvi/env/yolov5/bin/python", "/home/uvi/Traffic-Survey-v2.1/SetTimeFromDevice.py"],
            input="uvi123\n",  # ⚠️ 注意替换为你的实际sudo密码
            capture_output=True,
            text=True,
            timeout=30
        )
        print("[SetTimeFromDevice 输出]:")
        print(result.stdout)
        if result.stderr:
            print("[错误]:", result.stderr)
        return True
    except Exception as e:
        print(f"[同步异常]: {e}; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return False
