import subprocess
from datetime import datetime
import time

def ResetTime(ntp_server_ip="114.118.7.161"):
    # 由于硬编码密码不安全，这里仅作为示例保留，实际使用时应移除或替换
    password = "uvi123"
    command = ["sudo", "-S", "ntpdate", ntp_server_ip]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(input=password.encode())
        returncode = process.returncode
        if returncode != 0:
            print(f"ntpdate命令执行失败，错误信息：{stderr.decode()}")
            return False
        print(f"系统时间已同步到NTP服务器 {ntp_server_ip}")
        return True
    except Exception as e:
        print(f"执行ntpdate命令时发生错误: {e}")
        return False

def write_time_now(file_path, current_date):
    try:
        with open(file_path, 'a') as file:
            file.write(current_date + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def ReCheckTimeNow(file_path, ntp_server_ip="114.118.7.161"):
    try:
        if ResetTime(ntp_server_ip):
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"时间同步成功，当前时间为: {current_date}，且最新时间已更新到{file_path}")
            write_time_now(file_path, current_date)
            return True
        else:
            return False
    except Exception as e:
        # 捕获其他可能发生的异常
        print(f"时间同步过程中发生错误: {e}")
        return False

def CheckTimeEpoch(file_path, MaxEpoch=5, ntp_server_ip="114.118.7.161"):
    check_epoch = 0
    while check_epoch < MaxEpoch:
        if ReCheckTimeNow(file_path, ntp_server_ip):
            print(f"时间同步{check_epoch + 1}次，并同步成功")
            break
        check_epoch += 1
        print(f"时间同步{check_epoch}次，但没有同步成功，间隔10秒后继续同步时间!!!")
        time.sleep(10)
    else:
        # 如果循环结束仍未同步成功，则执行此块代码
        print(f"时间同步{MaxEpoch}次，但没有同步成功，IP:{ntp_server_ip}！！！")

if __name__ == '__main__':
    file_path = 'dateNow.txt'
    CheckTimeEpoch(file_path, MaxEpoch=10, ntp_server_ip="114.118.7.163")

