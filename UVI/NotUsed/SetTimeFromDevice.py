import os
import platform
import datetime
import ctypes
import sys
import re
import requests
import time
from typing import Union, Tuple


def check_device_status(device_code: str) -> Tuple[bool, Union[str, dict]]:
    """检查设备状态并返回时间字符串"""
    # 构建动态 URL（包含设备编码）
    url = f"http://222.219.137.122:19030/api/device/{device_code}"

    try:
        # 发送 GET 请求
        response = requests.get(url, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=10)

        if response.status_code == 200:
            # 尝试解析时间字符串
            time_str = response.text.strip()
            if re.match(r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$', time_str):
                print(f"[设备状态] 设备 {device_code} 在线 | 返回时间: {time_str}")
                return True, time_str
            else:
                print(f"[设备状态] 设备返回的时间格式无效: {time_str}")
                return False, None
        else:
            print(f"[设备状态] 请求失败 | 设备: {device_code} | 状态码: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"[设备状态] 请求异常 | 设备: {device_code} | 错误: {str(e)}")
        return False, None


def set_system_time(time_str: str, time_format: str = "%Y-%m-%d-%H-%M-%S") -> bool:
    """
    设置系统时间（跨平台实现）

    Args:
        time_str: 时间字符串
        time_format: 时间字符串的格式（默认为YYYY-MM-DD-HH-MM-SS）

    Returns:
        bool: 设置是否成功
    """
    system = platform.system()

    try:
        # 解析时间字符串为datetime对象
        new_time = datetime.datetime.strptime(time_str, time_format)
        print(f"[时间设置] 解析成功: {new_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Windows系统
        if system == "Windows":
            return _set_windows_time(new_time)

        # Linux或macOS系统
        elif system in ("Linux", "Darwin"):
            return _set_unix_time(new_time, system)

        # 不支持的系统
        else:
            print(f"[时间设置] ❌ 不支持的操作系统: {system}")
            return False

    except ValueError as e:
        print(f"[时间设置] ❌ 时间格式错误: {str(e)}")
        print(f"[时间设置] 输入的字符串: '{time_str}' | 预期格式: '{time_format}'")
        return False
    except Exception as e:
        print(f"[时间设置] ❌ 未知错误: {str(e)}")
        return False


def _set_windows_time(new_time: datetime.datetime) -> bool:
    """Windows系统时间设置实现"""
    try:
        # 检查管理员权限
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("[时间设置] ❌ 请使用管理员权限运行此脚本")
            return False

        # 导入Windows API函数
        kernel32 = ctypes.windll.kernel32

        # 定义SYSTEMTIME结构体
        class SYSTEMTIME(ctypes.Structure):
            _fields_ = [
                ("wYear", ctypes.c_ushort),
                ("wMonth", ctypes.c_ushort),
                ("wDayOfWeek", ctypes.c_ushort),
                ("wDay", ctypes.c_ushort),
                ("wHour", ctypes.c_ushort),
                ("wMinute", ctypes.c_ushort),
                ("wSecond", ctypes.c_ushort),
                ("wMilliseconds", ctypes.c_ushort)
            ]

        # 创建并填充SYSTEMTIME结构体
        st = SYSTEMTIME()
        st.wYear = new_time.year
        st.wMonth = new_time.month
        st.wDay = new_time.day
        st.wHour = new_time.hour
        st.wMinute = new_time.minute
        st.wSecond = new_time.second
        st.wMilliseconds = new_time.microsecond // 1000
        st.wDayOfWeek = new_time.weekday()  # 0=周一, 6=周日

        # 设置系统时间
        if kernel32.SetLocalTime(ctypes.byref(st)):
            print("[时间设置] ✅ Windows系统时间已成功更新")
            return True
        else:
            # 获取错误代码
            error_code = ctypes.windll.kernel32.GetLastError()
            print(f"[时间设置] ❌ Windows系统时间设置失败，错误代码: {error_code}")
            return False

    except Exception as e:
        print(f"[时间设置] ❌ Windows系统时间设置异常: {str(e)}")
        return False


# 修改函数签名
def _set_unix_time(new_time: datetime.datetime, system: str) -> bool:
    """Linux/macOS系统时间设置实现"""
    try:
        # 检查root权限
        if os.geteuid() != 0:
            print("[时间设置] ❌ 请使用root权限运行此脚本")
            return False

        if system == "Linux":
            date_str = new_time.strftime("%Y-%m-%d %H:%M:%S")
            os.system(f"date -s '{date_str}'")
            os.system("hwclock --systohc")
            print("[时间设置] ✅ Linux系统时间已成功更新")
            return True
        else:  # macOS
            date_str = new_time.strftime("%m%d%H%M%Y.%S")
            os.system(f"date {date_str}")
            print("[时间设置] ✅ macOS系统时间已成功更新")
            return True

    except Exception as e:
        print(f"[时间设置] ❌ Unix系统时间设置异常: {str(e)}")
        return False

def check_permissions():
    """检查当前运行环境权限"""
    system = platform.system()

    if system == "Windows":
        if not ctypes.windll.shell32.IsUserAnAdmin():
            return False, "请使用管理员权限运行此脚本"
    elif system in ("Linux", "Darwin"):
        if os.geteuid() != 0:
            return False, "请使用root/sudo权限运行此脚本"
    else:
        return False, f"不支持的操作系统: {system}"

    return True, "权限检查通过"


def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # 显示系统信息
    print("=" * 60)
    print(f"时间同步工具 | 系统: {platform.system()} {platform.release()}")
    print(f"当前系统时间: {get_current_time()}")

    # 检查权限
    has_permission, message = check_permissions()
    print(f"权限状态: {message}")

    device_id = "28254"

    # 获取设备时间
    print("\n" + "=" * 60)
    print("尝试从设备获取时间...")
    status, device_time = check_device_status(device_id)

    if not status:
        print("❌ 无法获取设备时间，脚本终止")
        sys.exit(1)

    # 确认操作
    print("\n" + "=" * 60)
    print(f"设备时间: {device_time}")
    print(f"当前系统时间: {get_current_time()}")

    if not has_permission:
        print("\n⚠️ 注意：您当前的权限不足，需要更高的权限才能修改系统时间")
        print("Windows: 请右键点击脚本，选择'以管理员身份运行'")
        print("Linux/macOS: 请使用sudo命令运行脚本")
        sys.exit(1)

    # confirm = input("\n是否要将系统时间更新为设备时间? (y/n): ").strip().lower()
    #
    # if confirm != 'y':
    #     print("操作已取消")
    #     sys.exit(0)

    # 设置系统时间
    print("\n" + "=" * 60)
    print("开始设置系统时间...")

    if set_system_time(device_time):
        # 显示更新后的时间
        time.sleep(1)  # 等待系统更新
        print(f"✅ 时间设置完成! 当前系统时间: {get_current_time()}")
    else:
        print("❌ 时间设置失败，请检查错误信息")