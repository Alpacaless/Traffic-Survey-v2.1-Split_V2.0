import os
import shutil
import UVI.SecondLevelFunc.send_file_to_server as send_file_to_server
import UVI.FirstLevelFunc.remove_line as remove_line

def resend_failed_files(possible_save_dirs):
    error_dir = "inference/error_logger"
    Num=0
    for log_file_name in os.listdir(error_dir):
        if not log_file_name.endswith('.txt'):
            continue

        date_str = log_file_name[:10]  # e.g. 2024-10-19
        log_file_path = os.path.join(error_dir, log_file_name)
        failed_files = log.read_read_log(log_file_path)
        line_i = 0
        if len(failed_files)>0:
            print(f"共有{len(failed_files)}辆车需要数据重传，请等待数据重传完成...")
        else:
            continue

        for file_name in failed_files:
            line_i += 1
            file_found = False
            

            # 依次尝试每个保存路径
            for base_dir in possible_save_dirs:
                candidate_path = os.path.join(base_dir, date_str, "txt", file_name)
                if os.path.exists(candidate_path):
                    print(f"✅ 找到文件: {file_name}，正在重传...")
                    if send_file_to_server.send_file_to_server(candidate_path):
                        print("重传成功")
                        Num = Num + 1
                        remove_line.remove_line(log_file_path, line_i)
                        line_i -= 1
                    else:
                        print(f"重传失败: {file_name}")
                    file_found = True
                    break  # 找到即终止查找

            if not file_found:
                print(f"文件{file_name}在以下目录中都未找到:")
                remove_line.remove_line(log_file_path, line_i)
                line_i -= 1
                for base_dir in possible_save_dirs:
                    print(f"  - {os.path.join(base_dir, date_str, 'txt')}")
    if Num>0:
        print(f"共{Num}辆车重发完成")
