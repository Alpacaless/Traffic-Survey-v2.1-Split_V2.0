import subprocess
from datetime import datetime, timedelta
import time

#def ResetTime(ntp_server_ip="114.118.7.161"):
#    # ntp_server_ip = "114.118.7.161"
#    password = "uvi123"
#    command = ["sudo", "-S","ntpdate", ntp_server_ip]
#    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    process.communicate(input=password.encode())
#    #subprocess.run(command, check=True)
#    print(f"系统时间已同步到NTP服务器 {ntp_server_ip}")

def ResetTime(ntp_server_ip="114.118.7.161"):
    password = "uvi123"  # 注意：硬编码密码不安全，建议移除或替换为更安全的方法
    command = ["sudo", "-S", "ntpdate", ntp_server_ip]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=password.encode())
    returncode = process.returncode
    if returncode != 0:
        # ntpdate命令执行失败，处理错误
        print(f"ntpdate命令执行失败，错误信息：{stderr.decode()}")
        # 可以选择抛出异常或返回错误码，以便上层函数处理
        # raise Exception("时间同步失败")
        return False
    print(f"系统时间已同步到NTP服务器 {ntp_server_ip}")
    return True

def write_time_now(file_path,current_date):
    try:
        with open(file_path, 'a') as file:
            file.write(current_date + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def ReCheckTimeNow(file_path,ntp_server_ip="114.118.7.161"):
    flag=False
    try:
        flag=ResetTime(ntp_server_ip)
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"时间同步成功，当前时间为: {current_date}，且最新时间已更新到{file_path}")
        flag=True
        write_time_now(file_path, current_date)
    except subprocess.CalledProcessError:
        # 同步时间失败，处理逻辑
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"时间同步失败，当前系统时间为: {current_date}，ip:{ntp_server_ip}")
    except Exception as e:
        # 其他异常处理
        print(f"时间同步失败，发生错误为: {e}")
    return flag

def CheckTimeEpoch(file_path,MaxEpoch=5,ntp_server_ip="114.118.7.161"):
    CheckEpoch=0
    flag=False
    while(CheckEpoch<MaxEpoch):
        flag=ReCheckTimeNow(file_path,ntp_server_ip)
        if flag:
            print(f"时间同步{CheckEpoch+1}次，并同步成功")
            break
        CheckEpoch=CheckEpoch+1
        print(f"时间同步{CheckEpoch}次，但没有同步成功，间隔10秒后继续同步时间!!!")
        time.sleep(10)

    if flag==False:
        print(f"时间同步{MaxEpoch}次，但没有同步成功，IP:{ntp_server_ip}！！！")

if __name__ == '__main__':
    file_path = 'dateNow.txt'
    CheckTimeEpoch(file_path,MaxEpoch=10,ntp_server_ip="114.118.7.163")

    
    


    
    
        
