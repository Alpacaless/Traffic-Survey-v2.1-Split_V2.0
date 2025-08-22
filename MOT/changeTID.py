import pandas as pd

def normalize_gt_ids(gt_path, output_path):
    df = pd.read_csv(gt_path, header=None)

    # 设置列名
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'ignore']

    # 按 class + id 组合为唯一识别符（你当前的问题正是 class+id）
    original_ids = df[['class', 'id']].astype(str).agg('-'.join, axis=1)

    # 给每个 class-id 对组合重新分配一个全局唯一ID
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(original_ids.unique()), start=1)}

    # 替换 id 列
    df['id'] = original_ids.map(id_map)

    # 保存新文件
    df.to_csv(output_path, header=False, index=False)
    print(f"已输出统一 ID 的 GT 文件：{output_path}")

# 用法
normalize_gt_ids('Bgt.txt', 'gt.txt')
