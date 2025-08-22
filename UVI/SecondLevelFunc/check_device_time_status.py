import UVI.FirstLevelFunc.is_valid_time_format as is_valid_time_format
import UVI.FirstLevelFunc.write_time_now as write_time_now
import UVI.FirstLevelFunc.read_last_line_of_file as read_last_line_of_file
import UVI.FirstLevelFunc.add_time as addtime
import UVI.ThirdLevelFunc.sync_device_time as sync_device_time
from datetime import datetime
import time
def check_device_time_status(start_time):
        TImeHMSNow=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 记录当前系统时间，每间隔10分钟记录一次，如果当前系统时间比上次时间更早，则不进行记录
        # 在代码执行过程中，保证时间更新不间断
        file_path = 'dateNow.txt'
            # Y-%m-%d_%H
        now = datetime.now()
        current_minute = now.minute-2#提前2分钟记录时间，防止因关机导致的读写错误
        if current_minute % 5 == 0:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            last_date = read_last_line_of_file.read_last_line_of_file(file_path)

            # 判断是否是时间格式，如果是时间格式，则进行时间更新；如果不是时间格式，则重新再写一行时间；
            if not is_valid_time_format.is_valid_time_format(last_date):
                try:
                    print(f"记录当前时间：{TImeHMSNow}")
                    with open(file_path, 'a') as file:
                        file.write('\n' + current_date + '\n')
                except Exception as e:
                    print(f"写入文件时发生错误：{e}；{TImeHMSNow}")

            # 如果是时间格式，则进行时间分析和更新；
            else:
                # 解析时间字符串为 datetime 对象
                time1 = datetime.strptime(current_date, "%Y-%m-%d %H:%M:%S")
                time2 = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
                time_difference = abs((time2 - time1).total_seconds())
                # 开发板重启之后，网络还没恢复，系统时间在去的某个时间，这时候可以不断尝试网络更新
                # 如果网络
                if (time1 < time2)&(time_difference>900):  # 如果current_date早于last_date，则说明current_date出现了问题，应该予以更新时间
                    print(f"{current_date} 早于 {last_date}，时间误差在15分钟以上;{TImeHMSNow}")
                    # 尝试调用服务器进行数据同步，并重新写时间
                    # 10s同步一次，持续同步MaxEpoch，如果时间同步成功，则写入file_path
                    # AutoReCheckTime.CheckTimeEpoch(file_path, MaxEpoch=10, ntp_server_ip="114.118.7.163")
                    sync_device_time.sync_device_time()  # 调用时间同步函数

                        # # 如果同步失败，则可能是断网了，可以通过时钟周期来进行代码校正
                        # new_last_date = read_last_line_of_file(file_path)
                        # if new_last_date == last_date:  # 数据没更新，则证明是失败的，则可以last_date记录时间+时钟周期最后更新时间
                        #     end_time = time.perf_counter()  # 获取当前时钟周期
                        #     elapsed_time = end_time - start_time
                        #     AddMinues = elapsed_time / 60
                        #     new_time_string = addtime(current_date, AddMinues)
                        #     write_time_now(file_path, new_time_string)
                        #
                        # start_time = time.perf_counter()  # 更新时钟周期

                elif time1 > time2:  # 如果current_date晚于last_date，说明时间大概率是正常的
                    # 计算时间差
                    time_difference = abs((time2 - time1).total_seconds())
                    if time_difference < 1800:
                        # 如果时间差在30分钟内，则证明时间是连续的，可以记录数据
                        write_time_now.write_time_now(file_path, current_date)
                        start_time = time.perf_counter()  # 获取当前时钟周期
                        print(f"current_date晚于last_date，并在半小时以内;{TImeHMSNow}")
                    else:  # 否则先同步数据
                            # AutoReCheckTime.CheckTimeEpoch(file_path, MaxEpoch=10, ntp_server_ip="114.118.7.161")
                            # 确实存在时间同步出现错误，但数据量比较少，可以通过时间同步进行修正
                            # 这里换个ip同步时间，如果同步时间失败，则这些数据可以舍弃
                        new_last_date = read_last_line_of_file.read_last_line_of_file(file_path)  # 检查时间同步是否成功，如果成功则更新时钟周期
                        if new_last_date != last_date:  # 数据更新了，则重新获取当前时钟周期
                            start_time = time.perf_counter()
                            print(f"current_date晚于last_date，时间误差在半小时以上;{TImeHMSNow}")
                else:
                    print(f"current_date：{current_date} ； last_date：{last_date};TImeHMSNow：{TImeHMSNow}")
                    write_time_now.write_time_now(file_path, current_date)

