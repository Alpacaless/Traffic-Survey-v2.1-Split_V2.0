import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.general import check_requirements
import UVI.SecondLevelFunc.run as run


# 执行分析功能
def analysisFunction(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查环境要求，排除tensorboard和thop
    
    run.run(**vars(opt))  # 调用run函数执行分析
