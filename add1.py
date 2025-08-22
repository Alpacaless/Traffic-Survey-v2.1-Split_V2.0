def process_gt_file(input_filename, output_filename):
    try:
        # 读取输入文件
        with open(input_filename, 'r') as f:
            lines = f.readlines()

        modified_lines = []
        for line_num, line in enumerate(lines, 1):  # 记录行号，从1开始
            # 去除首尾空白字符并按逗号分割成列表
            values = line.strip().split(',')

            # 检查元素数量
            if len(values) < 10:
                print(f"警告：第{line_num}行仅包含{len(values)}个值（需要10个），已跳过该行")
                modified_lines.append(line)  # 保留原始行
                continue

            try:
                # 获取第二个值（索引1）并尝试转换为数字
                second_value = values[1].strip()
                # 尝试转换为整数或浮点数
                if '.' in second_value:
                    num = float(second_value)
                else:
                    num = int(second_value)

                # 加1操作
                num += 1

                # 将结果转换回字符串并更新列表
                values[1] = str(num)

                # 将修改后的值重新用逗号组合成一行
                modified_line = ','.join(values) + '\n'
                modified_lines.append(modified_line)

            except ValueError:
                print(f"警告：第{line_num}行的第二个值'{second_value}'不是有效数字，已跳过该行")
                modified_lines.append(line)  # 保留原始行
            except IndexError as e:
                print(f"错误：第{line_num}行处理时发生索引错误: {str(e)}")
                modified_lines.append(line)  # 保留原始行
            except Exception as e:
                print(f"错误：第{line_num}行处理时发生错误: {str(e)}")
                modified_lines.append(line)  # 保留原始行

        # 写入输出文件
        with open(output_filename, 'w') as f:
            f.writelines(modified_lines)

        print(f"文件处理完成，已保存到 {output_filename}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_filename}")
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")


if __name__ == "__main__":
    # 输入文件名和输出文件名
    input_file = "Bgt.txt"
    output_file = "modified_gt.txt"

    # 调用函数处理文件
    process_gt_file(input_file, output_file)
