from datetime import datetime,timedelta,date
import subprocess
import time
import AutoReCheckTime
import os
import shutil
'''
代码用于开机重启，或断电重启后的时间更新，也就是sh文件重启

这个状态下的重启，有3个原因：
1  主动重启：由主动断电或自动Reboot导致的
特点：重启时间较为固定，在凌晨1:30——2:00
可以判断最新的txt时间是否在1:00——2:00之间，如果是的话，那么说明txt时间上是没问题的，直接在txt最后的时间上+10分钟即可

2  被动关机：开发板死机，重启时间不固定，也可以由主动断电重启
特点：关机时间比较随机，24小时内发生在次日凌晨1:00——2:00之间比较少，多数发生在早上4点以后
这种情况下，由于每天都有强制断电重启，可以强制将时间设置为次日凌晨1点50，然后进行时间同步
但这种情况下txt记录的时间是错的，可以通过日期+1,1:45的形式手动设置时间

3  手动Reboot
能够手动Reboot说明网络没有问题，只需自动更新时间即可

补充功能：自动删除45天前的所有文件夹的文件
'''


def read_last_line_of_file(file_path):
    """
    读取指定文件的最后一行非换行符字符串，并返回其作为字符串。
    如果文件为空，则返回空字符串。
    """
    last_line = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 直接读取所有行到列表中
            lines = file.readlines()
            # 如果文件不为空，则取最后一行并去除行尾的换行符或空白字符
            if lines:
                last_line = lines[-1].rstrip('\n\r ')
            # 注意：这里不需要再移动文件指针或读取字节串
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    return last_line



def adjust_datetime_string1(datetime_str,addBiasTime=5, date_format="%Y-%m-%d %H:%M:%S"):
    # 解析输入的日期时间字符串
    dt = datetime.strptime(datetime_str, date_format)
    
    new_datetime = dt + timedelta(minutes=addBiasTime)
    return new_datetime.strftime(date_format)

def adjust_Time(time_string):
    # 由于硬编码密码不安全，这里仅作为示例保留，实际使用时应移除或替换
    password = "uvi123\n"
    command = ["sudo", "-S", "date", "-s", time_string]  # -S 表示从 stdin 读取密码
    print("开始手动设置时间...")
    try:

        process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,  # 直接处理字符串而非字节
        bufsize=0
        )
        stdout, _ = process.communicate(input=password)
        print(stdout)
        returncode = process.returncode      
        if returncode != 0:
            print("手动设置时间命令执行失败")
            return False
        print(f"手动设置时间命令执行成功，{time_string}")
        return True
    except Exception as e:
        print(f"执行ntpdate命令时发生错误: {e}")
        return False



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



















########## 用了1

def adjust_datetime_string(datetime_str, date_format="%Y-%m-%d %H:%M:%S"):
    # 解析输入的日期时间字符串
    dt = datetime.strptime(datetime_str, date_format)
    
    # 提取时间部分
    current_time = dt.time()
    
    # 定义时间范围
    start_time = datetime.strptime("01:00:00", "%H:%M:%S").time()
    end_time = datetime.strptime("02:00:00", "%H:%M:%S").time()
    
    # 检查时间是否在范围1:00-2:00内
    if start_time <= current_time < end_time:
        # 在时间基础上累加10分钟
        new_datetime = dt + timedelta(minutes=10)
    else:
        # 将时间设置为次日01:45:00
        # 注意：这里我们需要创建一个新的日期时间对象，日期部分加一天，时间部分设置为01:45:00
        next_day = dt + timedelta(days=1)
        new_time = datetime.strptime("01:45:00", "%H:%M:%S").time()
        new_datetime = datetime.combine(next_day.date(), new_time)
    
    # 返回调整后的日期时间字符串
    return new_datetime.strftime(date_format)




if __name__ == '__main__':
    file_path = 'dateNow.txt'
    # current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S') #系统时间没有任何参考价值
    last_date = read_last_line_of_file(file_path)
    print(f"读取文档{file_path}中的时间为: {last_date}")
    adjusted_datetime_str = adjust_datetime_string1(last_date)

    # 先 手动调整时间为adjusted_datetime_str
    adjust_Time(adjusted_datetime_str)

    # 删除文件夹
    base_directory = r'E:\PHD\20250627MRNI\CODE\LATEST\Traffic-Survey-v2.1(2)\inference\output'  # 替换为你的文件夹路径
    delete_old_folders(base_directory)
    time.sleep(10)

    # 后系统自动更新时间
    # AutoReCheckTime.CheckTimeEpoch(file_path, MaxEpoch=5, ntp_server_ip="114.118.7.163")

