from datetime import datetime, timedelta


def adjust_datetime_string1(datetime_str,addBiasTime=5, date_format="%Y-%m-%d %H:%M:%S"):
    # 解析输入的日期时间字符串
    dt = datetime.strptime(datetime_str, date_format)
    
    new_datetime = dt + timedelta(minutes=addBiasTime)
    return new_datetime.strftime(date_format)