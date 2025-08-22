import os
def delete_errorlog():
    """
    删除 error_dir 中为空的错误日志文件。
    """
    error_dir = "inference/error_logger"
    deleted_count = 0
    if not os.path.exists(error_dir):
        print("❗ 错误日志目录不存在，无需删除空日志。")
        return

    for file_name in os.listdir(error_dir):
        file_path = os.path.join(error_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    print(f"🗑️ 删除空日志文件: {file_name}")
                    deleted_count += 1
            except Exception as e:
                print(f"⚠️ 无法删除 {file_name}: {e}")

    print(f"✅ 空日志清理完成，共删除 {deleted_count} 个文件。")
