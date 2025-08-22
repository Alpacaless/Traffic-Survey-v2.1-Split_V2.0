import os

class ReadLastLineOfFile:
    def read_last_line_of_file(file_path):
        """
        读取指定文件的最后一行非换行符字符串，并返回其作为字符串。
        如果文件为空，则返回空字符串。
        """
        last_line = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # 直接读取所有行到列表中
                lines = file.readlines()
                # 如果文件不为空，则取最后一行并去除行尾的换行符或空白字符
                if lines:
                    last_line = lines[-1].rstrip('\n\r ')
                # 注意：这里不需要再移动文件指针或读取字节串
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
        
        return last_line