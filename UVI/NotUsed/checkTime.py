import re
import subprocess
from datetime import datetime, timedelta
import time


def ResetTime():
    ntp_server_ip = "114.118.7.161"
    password = "uvi123"
    command = ["sudo", "-S","ntpdate", ntp_server_ip]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate(input=password.encode())
    #subprocess.run(command, check=True)
    print(f"系统时间已同步到NTP服务器 {ntp_server_ip}")

def read_last_line_of_file(file_path):
    last_line = ""
    try:
        with open(file_path, 'r') as file:
            for line in file:
                last_line = line.strip()  # 读取并去除首尾空白
            # 循环结束后，last_line将是文件的最后一行
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    return last_line

def extract_datetime_from_string(date_time_str):
    # 使用正则表达式匹配日期和时间
    date_time_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?')
    match = date_time_pattern.match(date_time_str)
    if match:
        year, month, day, hour, minute, second = match.groups()[:6]
        second = second if second else '00'  # 如果没有秒，则默认为00
        return year, month, day, hour, minute, second
    else:
        return None

def is_between_0000_and_0140(year, month, day, hour, minute, second):
    time_obj = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    start_of_day = datetime(int(year), int(month), int(day), 0, 0, 0)
    one_forty_am = start_of_day + timedelta(hours=1, minutes=40)
    return start_of_day <= time_obj <= one_forty_am

def write_time_now(file_path,current_date):
    try:
        with open(file_path, 'a') as file:
            file.write(current_date + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def handle_time_sync_failure(file_path, last_line):
    # 提取日期并加一天，然后设置时间
    new_date_string = add_one_day_to_date_string(last_line[:10])  # 假设时间部分不重要
    current_date = f"{new_date_string} 01:41:00"
    write_time_now(file_path, current_date)

def ReCheckTime(file_path, last_line):

    try:
        ResetTime()
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        write_time_now(file_path, current_date)
    except subprocess.CalledProcessError:
        # 同步时间失败，处理逻辑
        handle_time_sync_failure(file_path, last_line)
    except Exception as e:
        # 其他异常处理
        print(f"发生错误: {e}")
        handle_time_sync_failure(file_path, last_line)



def add_one_day_to_date_string(last_line, date_format="%Y-%m-%d"):
    date_string=last_line[0:10]
    # 将日期字符串转换为日期对象
    date_obj = datetime.strptime(date_string, date_format).date()
    # 日期加一天
    new_date_obj = date_obj + timedelta(days=1)
    # 将新的日期对象转换回字符串
    new_date_string = new_date_obj.strftime(date_format)
    return new_date_string

if __name__ == '__main__':
    file_path = 'dateNow.txt'
    last_line = read_last_line_of_file(file_path)
    time.sleep(2) 
    if last_line:
        date_time_parts = extract_datetime_from_string(last_line)
        if date_time_parts:
            year, month, day, hour, minute, second = date_time_parts
            print(f"日期: {year}-{month}-{day}, 时间: {hour}:{minute}:{second}")
            if is_between_0000_and_0140(year, month, day, hour, minute, second):
                print("时间在0点到1点40分之间。")#时间在0点到1点40分之间，证明日期是连续的，可以继续使用当前日期
            else:
                print("时间不在0点到1点40分之间。")
                #时间不在0点到1点40分之间，证明日期不是连续的，说明开发板在白天就已经断开
                #这种情况有两个可能：
                # 1. 开发板死机，次日断电重启恢复正常，此时开发板时间错开一日
                ReCheckTime(file_path,last_line)#此时，首先尝试同步时间，如果同步时间失败，则手动设置时间

                # 2. 系统没电，太阳能没电，这个情况下, 5G模块和开发板一起开机，大概率网络还没有恢复，
                # 可通过外网交互来判断，但时间间隔没办法判断
        else:
            print("无法从最后一行中提取日期和时间。")
    time.sleep(2) 
    last_line = read_last_line_of_file(file_path)
    print(last_line)

