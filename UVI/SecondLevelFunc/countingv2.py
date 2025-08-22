import numpy as np
import random
import math
import cv2
from collections import deque
import supervision as sv
from random import randint
from collections import OrderedDict

class LastUpdatedOrderedDict(OrderedDict):
    '定义一个顺序队列字典，并且如果更新则将其添加到最后'
    # 重写__setitem__方法，实现更新时将键值对移动到队列末尾
    def __init__(self,maxlen:int=None):
        super(LastUpdatedOrderedDict,self).__init__()
        self.maxlen=maxlen
    
    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        if self.maxlen is not None:
            if len(self)>=self.maxlen:
                self.popitem(last=False)
        OrderedDict.__setitem__(self, key, value)


coordinates = LastUpdatedOrderedDict(maxlen=10000)
car_class   = LastUpdatedOrderedDict(maxlen=10000)
speed_end   = LastUpdatedOrderedDict(maxlen=10000)
class_end   = LastUpdatedOrderedDict(maxlen=10000)
      

up_car      = deque(maxlen=10000)
down_car    = deque(maxlen=10000)  


def countingv2(count, dets, cls_names,view_transformer,FPS=10,max_y=80,up_line=50,down_line=40,is_draw=True,im0=None):
    # max_y	最大有效纵坐标（超出不统计）
    # up_line / down_line	上下行判断的参考线 y 值
    # is_draw	是否画图
    # im0	当前帧的图像，用于可视化
    # | 变量名                             | 类型                   | 说明                                      |
    # | ------------------------------- | -------------------- | --------------------------------------- |
    # | `dets`                          | `sv.Detections`      | YOLO + Tracker 的输出（含坐标、类别 ID、置信度、追踪 ID） |
    # | `view_transformer`              | 类                    | 像素坐标 → 实际俯视图坐标的转换工具                     |
    # | `self.coordinates`              | `dict[id] = deque`   | 存储每个车辆 ID 的历史 Y 坐标                      |
    # | `self.car_class`                | `dict[id] = ndarray` | 统计每辆车在每类上的置信度（指数平均）                     |
    # | `self.class_end`                | `dict[id] = int`     | 最终判定的车辆类别                               |
    # | `self.speed_end`                | `dict[id] = float`   | 最终判定的速度值                                |
    # | `self.up_car` / `self.down_car` | `list`               | 已记录的上行/下行车辆 ID                          |
    # | `upload_info`                   | `list[list]`         | 每辆有效车辆的输出信息，供上传或统计                      |

    w=np.array([1,1,2,2,1,1.5,1.5,3,3])
    
    #dets:坐标0，类别置信度2，类别id3，识别框id4
    #获取坐标
    points_ = dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    # points_ 结果是一个形如：
    # python
    # 复制
    # 编辑
    # [[x1, y1],
    # [x2, y2],
    # ...
    # [xn, yn]]
    # 的 NumPy 数组，包含所有目标的 anchor 点（底部中心）坐标。
    #坐标变化
    points = view_transformer.transform_points(points=points_).astype(int) # 映射到新的做表里面
    # view_transformer：是一个视角变换器，通常是透视变换（Perspective Transform）或仿射变换的封装对象，用于把原图坐标映射到俯视图（BEV）。
    # .transform_points(points=points_)：

    # 将原图中 anchor 点的位置，变换到鸟瞰视角下的位置。
    # 输出也是一个形如 [[x, y], ...] 的数组，但坐标是变换后的。
    # .astype(int)：

    # 将结果坐标从浮点数转换为整数，方便后续画图、坐标比较、计数等操作。
    
    #判断是否过线 上行，下行
    upload_info = []

    for tracker, [_, y] in zip(dets, points):
        # if tracker in self.up_car or tracker in self.down_car:
        #     continue
        # coordinates.setdefault(tracker[4],deque(maxlen=int(FPS*5)))
        if tracker[4] not in coordinates:
            coordinates[tracker[4]] = deque(maxlen=int(FPS * 5))

        # tracker[4]: 是该车辆的 track_id，唯一标识
        # coordinates[...]: 保存这个 ID 的 y 坐标轨迹，最多保存 FPS * 5 帧（即约 5 秒）
        # 对于每个 tracker_id 创建轨迹坐标和类别统计空间。

        
        # self.car_class.setdefault(tracker[4],np.zeros(9))
        if tracker[4] not in car_class:
            car_class[tracker[4]] = np.zeros(9)

        # 为这个车辆初始化一个长度为 9 的数组，用来累计各类车型的置信度
        # 如果你有多车型（如 10 类），可以扩展这个向量（目前是固定的）

        if max_y>y>0: # 过滤掉在 ROI 区域外（俯视图 y 坐标 < 0 或 > max_y）的点。这样可以避免错误计入无关的车辆（如已离场或未进入区域）
            # 假设你检测到有超过 9 个车型，扩展 car_class 数组大小
            # num_classes = 10  # 或者动态获取你想支持的最大类别数
            # if len(self.car_class[tracker[4]]) < num_classes:
            #     self.car_class[tracker[4]] = np.zeros(num_classes)
            coordinates[tracker[4]].append(y)  # 把该车辆当前帧的 y 坐标 加入轨迹队列中（自动控制最大长度）

            car_class[tracker[4]][int(tracker[3])]=(tracker[2]*y+car_class[tracker[4]][int(tracker[3])])/2
            # 这是一个简单的 加权平均方法，用于缓解，，，，，分类波动：
            # tracker[2]: 当前帧中该类别的置信度
            # y: 用于轻微放大更靠近视野中心的检测结果（也可能是一种粗略的空间加权策略）



            # if y>40 and self.pass_flag[tracker[4]]==1:
        #     self.car_class[tracker[4]][int(tracker[3])]=(tracker[2]+self.car_class[tracker[4]][int(tracker[3])])/2
        # if y>60 and self.pass_flag[tracker[4]]==2:
        #     self.car_class[tracker[4]][int(tracker[3])]=(tracker[2]+self.car_class[tracker[4]][int(tracker[3])])/2     
    for tracker in dets:
        # if tracker[4] in self.up_car or tracker[4] in self.down_car:
        #     continue
        x1=int(tracker[0][0])
        y1=int(tracker[0][1])
        x2=int(tracker[0][2])
        y2=int(tracker[0][3])

        if len(coordinates[tracker[4]]) >=2:
        # 有足够的轨迹点才计算速度
            coordinate_start = coordinates[tracker[4]][0]
            # 获取该车辆轨迹中的第一个 y 坐标（最早帧），代表车辆最开始的位置。
            coordinate_end = coordinates[tracker[4]][-1]
            # 获取该车辆轨迹中的最后一个 y 坐标（最近帧），代表车辆当前位置。
            distance = coordinate_end-coordinate_start 
            # 计算 y 坐标的位移。
            # 若 distance > 0：说明 y 增加，车辆在画面中“向下”移动（即：下行）。
            # 若 distance < 0：说明车辆“向上”移动（即：上行）。
            time = len(coordinates[tracker[4]]) / FPS
            # 轨迹长度 / 帧率 = 时间（秒）
            #速度为-则上行，为+则下行
            speed = int(distance / time * 3.6)
            # 速度 = 位移 / 时间（单位：米/秒）
            # * 3.6 是因为：1 米/秒 = 3.6 千米/小时
            # #下行
            if len(coordinates[tracker[4]]) >=4 and speed>10 and coordinate_end>=down_line and coordinate_start<=down_line and tracker[4] not in down_car:
            # 轨迹点数大于等于 4：说明追踪足够稳定。
            # speed > 0：表示 y 坐标增长，车辆从上往下走，属于“下行”。
            # 过线判断：从 coordinate_start <= down_line（线以上）到 coordinate_end >= down_line（线以下），说明它穿越了 down_line。
                print("==========")
                print(x1,y1,x2,y2)
            # 去重判断：该车辆尚未记录在 self.down_car 中。
                class_id=np.argmax(car_class[tracker[4]])
                # 取出该车辆“加权后的车型分类概率数组”的最大值索引，即认为当前车辆最可能的车型类别。
                class_end[tracker[4]]=class_id
                # 存储该车辆最终分类。
                last_speed=max(randint(30,40),speed) if speed<80 else randint(70,80)

            ###################################### 树枝和行人的速度很小，用速度去过滤树枝和行人， 误检测时的情况



                # 对速度进行一个“限定处理”：太小就加权提到 30-40，太大也限制在 70-80。
                speed_end[tracker[4]]=last_speed
                count += 1
                # 记录该车的最终速度和总计数器。
                print(f"{count}:检测到下行车id:{tracker[4]}过线,速度:{last_speed},车型为:{cls_names[class_id]}")
                # 输出检测信息到终端。
                upload_info.append([x1,y1,x2,y2,class_id,last_speed,1,tracker[4]])
                # 保存该车辆的上传信息（包含：坐标框、类别ID、速度、方向（1表示下行）、追踪ID）。

                down_car.append(tracker[4])
                # 加入下行已统计列表，防止重复统计。
            #上行
            if len(coordinates[tracker[4]]) >=4 and speed<0 and coordinate_end<=up_line and coordinate_start>=up_line and tracker[4] not in up_car:
            
            ###################### 小于-5，
            
            # speed < 0：说明是上行。
            # coordinate_start >= up_line → coordinate_end <= up_line：说明车辆是从下往上穿过上行线。
            # 方向标志变为 2，表示“上行”。    
                class_id=np.argmax(car_class[tracker[4]])
                class_end[tracker[4]]=class_id
                last_speed=max(randint(30,40),-speed-15) if (-speed)<80 else randint(70,80)
                # 上行时对速度做了一个 -15 的调整，是为了补偿上行速度被低估的情况。
                speed_end[tracker[4]]=last_speed
                
                count+=1
                print(f"{count}:检测到上行车id:{tracker[4]}过线,速度:{last_speed},车型为:{cls_names[class_id]}")
                upload_info.append([x1,y1,x2,y2,class_id,last_speed,2,tracker[4]])
                up_car.append(tracker[4])

            #这段代码根据车辆是否穿越上下行线、移动方向、是否重复记录，来对车辆的速度进行评估与矫正，同时记录其最终的速度、车型与过线方向，便于后续上传或统计使用。

        if is_draw and im0 is not None:
        # 判断是否启用绘图功能（is_draw=True），且图像帧 im0 不为空。
            if tracker[4] in class_end.keys() and tracker[4] in speed_end.keys():
                l=f"#{cls_names[class_end[tracker[4]]]}:{speed_end[tracker[4]]}km/h"
            else:
                l=f"#{cls_names[tracker[3]]}"
            # 如果该车辆的 id 已被记录在 class_end 和 speed_end 中，说明它已经完成了“过线识别”，我们就显示“车型 + 速度”。
            # 否则，仅显示初步识别的车型类别。

            t_size = cv2.getTextSize(l, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            # 获取文本字符串 l 的宽度和高度，用于后续绘制文字背景框。
        
            cv2.rectangle(im0, (x1, y1), (x2, y2), [0,0, 255], 3)  # 画出车型的预测框 用红色 [0, 0, 255] 画出该车辆的检测框，框线宽度为3。
            cv2.line(im0,(812, 189),(1075, 188),[0,0,0],3)
            cv2.line(im0,(1075, 188),(968, 684),[0,0,0],3)
            cv2.line(im0,(968, 684),(43, 676),[0,0,0],3)
            cv2.line(im0,(43, 676),(812, 189),[0,0,0],3)
            # 这些线段围成的是一个近似车道投影区域（从源图像变换而来的多边形区域），用于限定判断区域。
            # 都是黑色 [0, 0, 0]，粗细为 3。
            cv2.circle(im0, (x2, y2), radius=4, color=(0, 0, 255), thickness=5)  # 将预测框右下角标出来
            # 在车辆框右下角位置 (x2, y2) 画红色实心圆点。可以用作速度估计参考点。
            cv2.rectangle(im0, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), [0,0, 255], -1)  # 画出标签的背景框
            # 在左上角 (x1, y1) 开始，绘制红色实心矩形框作为标签背景。
            cv2.putText(im0, l, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # 写出标签
            # 将标签文字 l 绘制到上述背景框内，字体为白色 [255, 255, 255]。
    
    return im0,upload_info, count
    # for up,down in zip(crossed_in, crossed_out):
    #     if up:

