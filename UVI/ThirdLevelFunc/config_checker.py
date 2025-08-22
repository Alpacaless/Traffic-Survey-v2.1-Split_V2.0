import sys
sys.path.insert(0, './yolov5')
import time
from datetime import datetime

import UVI.FirstLevelFunc.count_threads_by_name as count_threads_by_name
import UVI.ThirdLevelFunc.sync_device_time as sync_device_time
import UVI.FirstLevelFunc.WriteNowTime as WriteNowTime
import UVI.SecondLevelFunc.get_current_config as get_current_config
import UVI.ThirdLevelFunc.start_run_thread as start_run_thread


import UVI.SecondLevelFunc.check_device_time_status as check_device_time_status
import UVI.SecondLevelFunc.kill_termination_PID as kill_termination_PID


def config_checker(need_restart = False):
    current_source, current_weights, current_save_dir = get_current_config.get_current_config()
    stop_run_flag = False
    current_run_thread = None

    start_time = time.perf_counter() #开始记录时间的时钟周期
    time.sleep(90)
    while True:
        WriteNowTime.WriteNowTime()
        CodeNumber = count_threads_by_name.count_threads_by_name('MainCode')
        if CodeNumber < 3:
            #如果总线程数少于3，那么说明有一个线程没有执行，则代码重启
            print(f'视频流线程监测到总线程数不足，只有：{CodeNumber}，代码重启')
            kill_termination_PID.kill_termination_PID()
        
        now = datetime.now()
        current_minute = now.minute-2#提前3分钟记录时间，防止因关机导致的读写错误
        # 判断是否是时间格式，如果是时间格式，则进行时间更新；如果不是时间格式，则重新再写一行时间；
        if current_minute % 30==0:
            sync_device_time.sync_device_time() #30分钟自动同步时钟一次
        
        check_device_time_status.check_device_time_status(start_time)
        new_source, new_weights, new_save_dir = get_current_config.get_current_config()
        
        # 1. 配置变了
        if new_source != current_source or new_weights != current_weights or new_save_dir != current_save_dir:
            print(f"🌀 检测到配置变化，从 {current_source} 切换到 {new_source}")
            print(f"模型从 {current_weights} 切换到 {new_weights}")
            print(f"输出目录切换到 {new_save_dir}")
            print("🌀 检测到配置变化，准备重启")
            need_restart = True

        # 2. run() 自然结束或超时退出
        restart_count = 0
        if stop_run_flag and (not current_run_thread or not current_run_thread.is_alive()):
            print("🔄 run() 已结束，准备重启")
            restart_count += 1
            if restart_count > 2:
                print("❌ run() 重启超过3次仍失败，程序退出")
            need_restart = True

        if need_restart:
            if current_run_thread and current_run_thread.is_alive():
                print("⏳ 发送退出信号给旧线程...")
                stop_run_flag = True  # ① 通知退出
                kill_termination_PID.kill_termination_PID()

                '''
                # 旧代码，推出线程信号，代码存在的问题：只推出了当前线程，未正式推出程序，线程重新启动不会重新加载视频流
                print("⏳ 发送退出信号给旧线程...")
                stop_run_flag = True  # ① 通知退出
                os._exit(0)  # 强制退出所有线程，立即关闭
                current_run_thread.join()  # ② 等待退出
                print("✅ 旧线程已退出")
                '''

            stop_run_flag = False  # ③ 重置退出标志，准备启动新线程

            # 更新配置
            opt.source = new_source
            opt.weights = new_weights
            opt.project = new_save_dir
            # 使用现有 opt 重启（不需要变更 source/weights）
            start_run_thread.start_run_thread(opt)

            # 记录最新配置
            current_source, current_weights, current_save_dir = new_source, new_weights, new_save_dir
            print("🚀 线程重启完成")

        time.sleep(50)
