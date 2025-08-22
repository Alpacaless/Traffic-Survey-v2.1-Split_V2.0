def clean_data(file_path):
    """
    清洗输入文件数据
    功能：检查每行是否包含10个数值，过滤空行、格式错误行和非数字行
    参数：file_path - 输入文件路径（如"Bgt.txt"或"track.txt"）
    返回：清洗后的列表，每个元素为一行数据的浮点数列表
    """
    clean_lines = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行
            if not line:
                print(f"跳过空行：第{line_num}行")
                continue
            # 按逗号分割元素
            values = line.split(',')
            # 检查元素数量是否为10个
            if len(values) != 10:
                print(f"格式错误：第{line_num}行仅包含{len(values)}个元素（需10个），已跳过")
                continue
            # 尝试转换为浮点数（确保可计算）
            try:
                numeric_values = [float(v) for v in values]
                clean_lines.append(numeric_values)
            except ValueError:
                print(f"格式错误：第{line_num}行包含非数字值，已跳过")
                continue
    print(f"文件{file_path}清洗完成，有效行数：{len(clean_lines)}")
    return clean_lines


def calculate_iou(gt_box, track_box):
    """
    计算两个目标框的交并比（IOU）
    参数：
        gt_box - 清洗后的gt数据行（列表）
        track_box - 清洗后的track数据行（列表）
    返回：IOU值（0~1），异常情况返回0并提示
    说明：假设每行第3-6个值为x, y, width, height（索引2-5）
    """
    # 提取目标框坐标和宽高（x,y为左上角坐标）
    gt_x, gt_y, gt_w, gt_h = gt_box[2], gt_box[3], gt_box[4], gt_box[5]
    tr_x, tr_y, tr_w, tr_h = track_box[2], track_box[3], track_box[4], track_box[5]

    # 检查宽高是否有效（必须大于0）
    if gt_w <= 0 or gt_h <= 0:
        print(f"GT框异常：宽高为0（ID={int(gt_box[1])}）")
        return 0.0
    if tr_w <= 0 or tr_h <= 0:
        print(f"Track框异常：宽高为0（ID={int(track_box[1])}）")
        return 0.0

    # 计算目标框的对角坐标（左上角→右下角）
    gt_left, gt_right = gt_x, gt_x + gt_w
    gt_top, gt_bottom = gt_y, gt_y + gt_h
    tr_left, tr_right = tr_x, tr_x + tr_w
    tr_top, tr_bottom = tr_y, tr_y + tr_h

    # 计算交集区域
    inter_left = max(gt_left, tr_left)
    inter_right = min(gt_right, tr_right)
    inter_top = max(gt_top, tr_top)
    inter_bottom = min(gt_bottom, tr_bottom)
    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height

    # 计算并集区域
    gt_area = gt_w * gt_h
    tr_area = tr_w * tr_h
    union_area = gt_area + tr_area - inter_area

    # 避免并集为0（理论上不会发生，因已过滤宽高≤0的情况）
    if union_area == 0:
        print(f"并集为0（ID={int(gt_box[1])}）")
        return 0.0

    return inter_area / union_area


def match_and_calculate(gt_data, track_data, output_file="results.txt"):
    """
    按目标ID匹配gt和track数据，计算IOU并保存结果
    参数：
        gt_data - 清洗后的gt数据
        track_data - 清洗后的track数据
        output_file - 结果保存路径
    返回：无（结果保存到文件）
    """
    # 按目标ID分组（假设第2个值为ID，索引1）
    gt_dict = {int(gt[1]): gt for gt in gt_data}  # key: ID, value: 数据行
    track_dict = {int(tr[1]): tr for tr in track_data}

    # 计算匹配的IOU
    results = []
    matched_ids = set(gt_dict.keys()) & set(track_dict.keys())
    unmatched_gt = set(gt_dict.keys()) - matched_ids
    unmatched_track = set(track_dict.keys()) - matched_ids

    # 输出未匹配的ID
    if unmatched_gt:
        print(f"未在track中找到的GT ID：{sorted(unmatched_gt)}")
    if unmatched_track:
        print(f"未在GT中找到的Track ID：{sorted(unmatched_track)}")

    # 计算匹配ID的IOU
    for target_id in sorted(matched_ids):
        iou = calculate_iou(gt_dict[target_id], track_dict[target_id])
        results.append(f"ID: {target_id}, IOU: {iou:.4f}")
        print(results[-1])  # 实时打印

    # 保存结果到文件
    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    print(f"计算完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    # 主流程：清洗数据→匹配计算→保存结果
    gt_clean = clean_data("gt.txt")
    track_clean = clean_data("track.txt")
    if gt_clean and track_clean:  # 确保清洗后有有效数据
        match_and_calculate(gt_clean, track_clean)
    else:
        print("无有效数据可计算，请检查输入文件")