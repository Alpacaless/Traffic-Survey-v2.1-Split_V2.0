import os


def get_python_processes():
    """获取所有Python进程信息"""
    processes = []
    for entry in os.scandir('proc'):
        if entry.is_dir() and entry.name.isdigit():
            pid = entry.name
            try:
                with open(f'proc/{pid}/status', 'r') as f:
                    for line in f:
                        if line.startswith('Name:'):
                            process_name = line.split(':')[1].strip().lower()
                            if 'python' in process_name:
                                cmdline_path = f'proc/{pid}/cmdline'
                                if os.path.exists(cmdline_path)and len(process_name)==6:
                                    with open(cmdline_path, 'r') as cmd_file:
                                        cmdline = cmd_file.read().replace('\x00', ' ').strip()
                                        processes.append({
                                            "pid": pid,
                                            "name": process_name,
                                            "command": cmdline
                                        })
                                break
            except (IOError, PermissionError):
                continue
    return processes