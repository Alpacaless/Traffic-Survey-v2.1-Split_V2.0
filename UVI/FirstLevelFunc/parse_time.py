from datetime import datetime
def parse_time(time_str):
    """处理24小时制时间字符串"""
    if time_str == "24:00:00":
        return datetime.strptime("23:59:59", "%H:%M:%S").time()
    return datetime.strptime(time_str, "%H:%M:%S").time()

