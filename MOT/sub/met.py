import motmetrics as mm
import os
import numpy as np
import pandas as pd

def read_mot_file(filename):
    """读取 MOT 格式的 txt 文件"""
    # 读取文件内容，列名需手动设置
    columns = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Score', 'ClassId', 'Vis', 'Ignored']
    df = pd.read_csv(filename, header=None, names=columns)
    df = df[df['Ignored'] == 0]  # 只保留需要评估的目标
    return df

def mot_evaluate(gt_file, pred_file):
    # 加载 Ground Truth 和 Track 文件
    gt = read_mot_file(gt_file)
    pred = read_mot_file(pred_file)

    # 初始化 MOT Evaluator
    acc = mm.MOTAccumulator(auto_id=True)

    # 获取所有帧
    frames = sorted(gt['FrameId'].unique())

    for frame in frames:
        gt_frame = gt[gt['FrameId'] == frame]
        pred_frame = pred[pred['FrameId'] == frame]

        gt_ids = gt_frame['Id'].values
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values

        pred_ids = pred_frame['Id'].values
        pred_boxes = pred_frame[['X', 'Y', 'Width', 'Height']].values

        # 计算两组框的距离 (IoU)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1', 'idfp', 'idfn', 'precision', 'recall', 'num_matches', 'num_objects', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches'], name='overall')
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={
            'mota': 'MOTA',
            'idf1': 'IDF1',
            'num_switches': 'IDSW',
            'num_false_positives': 'FP',
            'num_misses': 'FN'
        }
    )

    print(strsummary)

# 使用路径运行（请确保路径正确）
mot_evaluate("gt.txt", "track.txt")
