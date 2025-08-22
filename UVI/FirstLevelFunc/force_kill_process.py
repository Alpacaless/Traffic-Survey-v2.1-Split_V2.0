import subprocess


def force_kill_process(pid_source):
    print("开始终止PID为",pid_source,"的进程")
    pid=int(pid_source)    
                    
    try:
        # 验证PID是否为有效整数
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("PID必须是正整数")
        
        # 执行强制终止命令（适用于Linux/macOS）
        result = subprocess.run(
            ["kill", "-9", str(pid)],  # "-9"表示发送SIGKILL信号（强制终止）
            check=True,                # 命令执行失败时抛出异常
            capture_output=True,       # 捕获输出
            text=True                  # 以文本模式处理输出
        )
        
        # 输出成功信息
        print(f"成功强制终止PID为 {pid} 的进程")
        return True
    
    except subprocess.CalledProcessError as e:
        # 命令执行失败（如进程不存在、无权限）
        print(f"终止失败：{e.stderr.strip()}")
        return False
    except ValueError as e:
        # PID格式错误
        print(f"参数错误：{e}")
        return False

