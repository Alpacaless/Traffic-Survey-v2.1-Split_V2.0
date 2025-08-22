import subprocess

def adjust_Time(time_string):
    # 由于硬编码密码不安全，这里仅作为示例保留，实际使用时应移除或替换
    password = "uvi123\n"
    command = ["sudo", "-S", "date", "-s", time_string]  # -S 表示从 stdin 读取密码
    print("开始手动设置时间...")
    try:

        process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,  # 直接处理字符串而非字节
        bufsize=0
        )
        stdout, _ = process.communicate(input=password)
        print(stdout)
        returncode = process.returncode      
        if returncode != 0:
            print("手动设置时间命令执行失败")
            return False
        print(f"手动设置时间命令执行成功，{time_string}")
        return True
    except Exception as e:
        print(f"执行ntpdate命令时发生错误: {e}")
        return False
