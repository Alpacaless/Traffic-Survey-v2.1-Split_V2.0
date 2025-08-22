
def write_time_now(file_path,current_date):
    try:
        with open(file_path, 'a') as file:
            file.write(current_date + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")
