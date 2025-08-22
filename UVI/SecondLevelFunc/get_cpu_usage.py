import os
from UVI.FirstLevelFunc.get_python_processes import get_python_processes

def get_cpu_usage():
    """计算并返回CPU占用最高的Python进程PID"""
    python_processes =  get_python_processes()
    if not python_processes:
        print("错误：未找到运行的Python进程")
        return None
    
    process_cpu = {}
    for proc in python_processes:
        pid = proc['pid']
        try:
            with open(f'/proc/{pid}/stat', 'r') as f:   
                stat = f.read().split()
                # 解析用户态和内核态CPU时间（第14和15列）
                utime = int(stat[13])
                stime = int(stat[14])
                process_cpu[pid] = utime + stime
        except (IOError, PermissionError):
            continue
    
    # 获取系统总CPU时间
    try:
        with open('/proc/stat', 'r') as f:
            stat = f.read().split('\n')[0].split()
            total_cpu_time = sum(map(int, stat[1:]))
    except (IOError, PermissionError):
        print("错误：无法读取系统CPU时间")
        return None
    
    # 计算并排序CPU占用率
    max_cpu_pid = None
    max_cpu_usage = 0
    for pid, p_time in process_cpu.items():
        cpu_usage = (p_time / total_cpu_time) * 100
        if cpu_usage > max_cpu_usage:
            max_cpu_usage = cpu_usage
            max_cpu_pid = pid

    return max_cpu_pid
