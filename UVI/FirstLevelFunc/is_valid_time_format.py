from datetime import datetime, timedelta


def is_valid_time_format(time_str):   # 1级函数
    time_format = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(time_str,time_format)
        return True
    except ValueError:
        return False