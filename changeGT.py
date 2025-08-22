def process_gt_file(input_filename, output_filename):
    # 定义字符串到数字的映射字典
    category_mapping = {
        "Motorcycle": 0,
        "Car": 1,
        "Bus": 2,
        "Tractor": 3,
        "L_truck": 4,
        "XL_truck": 5,
        "XXL_truck": 6,
        "XXXl_truck": 7,
        "Container car": 8
    }

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
                # 1. 修改第7个值（索引为6）：-1改为1
                if values[6].strip() == '-1':
                    values[6] = '1'

                # 2. 交换第8个值（索引7）和第10个值（索引9）
                values[7], values[9] = values[9], values[7]

                # 3. 将交换后的第8个值（索引7）从字符串替换为对应数字
                # 去除可能的空格
                value_str = values[7].strip()
                if value_str in category_mapping:
                    values[7] = str(category_mapping[value_str])
                else:
                    print(f"警告：第{line_num}行的第8个值'{value_str}'不在映射字典中，未替换")

                # 将修改后的值重新用逗号组合成一行
                modified_line = ','.join(values) + '\n'
                modified_lines.append(modified_line)

            except IndexError as e:
                print(f"错误：第{line_num}行处理时发生索引错误: {str(e)}，元素数量为{len(values)}")
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
    output_file = "processed_gt.txt"

    # 调用函数处理文件
    process_gt_file(input_file, output_file)
