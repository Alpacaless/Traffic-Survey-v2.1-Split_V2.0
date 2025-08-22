import UVI.SecondLevelFunc.get_cpu_usage as get_cpu_usage
import UVI.FirstLevelFunc.force_kill_process as force_kill_process    

def kill_termination_PID():
    highest_pid = get_cpu_usage.get_cpu_usage()
    print("+++++++++++++++++++++++++highest_pid+++++++++++++++++++++++++",highest_pid)
    print(f"准备终止PID {highest_pid} 的进程")
    force_kill_process.force_kill_process(highest_pid)
