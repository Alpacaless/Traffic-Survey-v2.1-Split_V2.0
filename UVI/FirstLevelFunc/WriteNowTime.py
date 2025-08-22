from datetime import datetime
import UVI.FirstLevelFunc.write_time_now as write_time_now

def WriteNowTime():
    file_path = 'dateNow.txt'
    TImeHMSNow=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.now()
    current_minute = now.minute-2#提前2分钟记录时间，防止因关机导致的读写错误
    # 判断是否是时间格式，如果是时间格式，则进行时间更新；如果不是时间格式，则重新再写一行时间；
    if current_minute % 3==0:
        try:
            with open(file_path, 'a') as file:
                write_time_now.write_time_now(file_path, TImeHMSNow)
        except Exception as e:
            print(f"时间写入发生错误：{e}:{TImeHMSNow}")

