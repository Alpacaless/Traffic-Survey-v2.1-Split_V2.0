from datetime import datetime, timedelta


# 直接使用的时候，用的是这个函数
def addtime(time_string,AddMinues):
    time_obj = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    # 将一个表示时间的字符串转换为 datetime 对象
    
    time_obj += timedelta(minutes=AddMinues)
    # timedelta 是 Python datetime 模块中的一个类，表示“时间间隔”。
    # 这里通过 minutes=AddMinues 创建一个“若干分钟”的时间差对象。
    # 将这个时间间隔 加到原来的时间对象 time_obj 上。
    # 例如：AddMinues = 15，就相当于 timedelta(minutes=15)
    # time_obj = datetime(2025, 7, 4, 20, 30, 0)
    # 输出: 2025-07-04 20:45:00
    
    new_time_string = time_obj.strftime("%Y-%m-%d %H:%M:%S")
    # 将 datetime 对象转换回时间字符串

    return new_time_string