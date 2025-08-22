import os



def clean_file_keep_last_lines(file_path, keep_lines=100):
    try:
        if not os.path.exists(file_path):
            print(f"{os.path.basename(file_path)} 不存在，无需清理。")
            return  

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) > keep_lines:
            lines = lines[-keep_lines:]  # 只保留最后 keep_lines 行

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"✅ {os.path.basename(file_path)} 清理完成，只保留最后 {keep_lines} 行。")
        else:
            print(f"✅ {os.path.basename(file_path)} 行数少于 {keep_lines}，无需清理。")
    except Exception as e:
        print(f"⚠️ 清理 {os.path.basename(file_path)} 出错: {e}")

