from datetime import datetime

def is_within_time_range():
    """检查当前时间是否在18点前"""
    current_time = datetime.now().time()
    return  current_time.hour < 19 and current_time.hour > 9  # 小时数小于18表示在18点前
