import time
from datetime import datetime


import UVI.ThirdLevelFunc.resend_failed_files as resend_failed_files
import UVI.FirstLevelFunc.check_device_status as check_device_status
import UVI.FirstLevelFunc.count_threads_by_name as count_threads_by_name
import UVI.ThirdLevelFunc.sync_device_time as sync_device_time
import UVI.SecondLevelFunc.kill_termination_PID as kill_termination_PID
import UVI.FirstLevelFunc.load_config as load_config


def start_check_device_status():
    configs = load_config.load_config()  # 获取设备编号
    device_code2 = configs['device_code1']  # 获取设备编号
    last_sync_minute = -1  # 初始化上一次同步的分钟值
    TImeHMSNow=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time.sleep(90)
    while True:
        time.sleep(25)  # 睡眠25秒
        CodeNumber=count_threads_by_name.count_threads_by_name('MainCode')
        if CodeNumber < 3:
            #如果总线程数少于3，那么说明有一个线程没有执行，则代码重启
            print(f'状态信号线程监测到总线程数不足，只有：{CodeNumber}，代码重启')
            kill_termination_PID.kill_termination_PID()
        current_minute = int(datetime.now().strftime('%M'))  # 获取当前分钟数
        # current_hour = datetime.now().strftime('%H')  # 如果需要按小时判断也可以用

        check_device_status.check_device_status(device_code2)  # 检查设备状态
        if current_minute % 2 == 0 and current_minute != last_sync_minute:
            print(f"Resend Failed Data Now!{TImeHMSNow}")
            resend_failed_files.resend_failed_files(['inference/output/night', 'inference/output', 'inference/output/daytime'])  # 调用重发失败文件的函数
        if current_minute % 20 == 0 and current_minute != last_sync_minute:
            # 更新上次同步分钟
            last_sync_minute = current_minute
            state=sync_device_time.sync_device_time()  # 调用时间同步函数
            print(f".sync_device_time()的时间同步完成，返回为{state}")
