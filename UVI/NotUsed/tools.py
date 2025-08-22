import math  # 数学公式模块
import random
import torch
import os  # 与操作系统进行交互的模块
import cv2  # opencv库
from queue import Queue
import matplotlib  # matplotlib模块
import matplotlib.pyplot as plt  # matplotlib画图模块
import numpy as np  # numpy矩阵处理函数库
from collections import defaultdict, deque,OrderedDict,Counter
import supervision as sv
from random import randint
class Colors:
    """
        函数功能：这是一个颜色类，用于选择相应的颜色，比如画框线的颜色，字体颜色等等。
    """

    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]  
        # 将hex列表中所有hex格式(十六进制)的颜色转换rgb格式的颜色
        # 'FF3838' → RGB(255, 56, 56)
        self.n = len(self.palette)  # 颜色个数

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]  
        # 根据输入的index 选择对应的rgb颜色
        return (c[2], c[1], c[0]) if bgr else c  # 返回选择的颜色 默认是rgb

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        # 把 '#FF3838' 这样的十六进制颜色转成 RGB 三元组。
        # 用 int('FF', 16) 将字符串 FF 转为十进制 255。


class EstimateSpeed:
    """
        函数功能：主要是用来估计车辆的速度
    """

    def __init__(self, point):
        self.speed_queue = Queue(maxsize=10)  
        # 存储跟踪车辆前10帧的位置队列
        if point is not None:
            self.speed_queue.put(point)
            # 初始化时将首帧位置入队：

        self.state = 0  # 当前跟踪目标的速度估计状态，
        # 0：跟踪目标未满足计算状态 
        # 3：跟踪目标满足3帧计算状态 
        # 5：跟踪目标满足5帧计算状态
        # 10：跟踪目标满足10帧计算状态 
        # 1：跟踪目标处于被删除状态
        self.down_speed_ref = 15 / 87 # 下行时，参考线附近15米对应170个像素距离
        self.up_speed_ref = 10 / 124  # 上行时，参考线附近10米对应247个像素距离
        self.box_speed = 0  # 车辆的初始速度

    def update(self, point):
        """
        更新当前跟踪目标的跟踪点信息

        Parameters
        ----------
        point : 当前需要更新的点信息
        """
        # 根据当前队列的状态更新队列成员
        if point is not None and self.speed_queue.qsize() < 10:
            self.speed_queue.put(point)
        elif point is not None and self.speed_queue.qsize() == 10:
            self.speed_queue.get()
            self.speed_queue.put(point)
        else:
            pass

        # 根据当前队列的大小更新速度计算状态
        if 3 <= self.speed_queue.qsize() < 5: 
            #.qsize 当前队列中的点数量
            self.state = 3
        elif 5 <= self.speed_queue.qsize() < 10:
            self.state = 5
        elif self.speed_queue.qsize() == 10:
            self.state = 10
        else:
            self.state = 0
            # 如果队列为空，说明目标未被跟踪，状态置为0
        self.state=self.speed_queue.qsize()


    def estimate_speed(self, box, pass_flag):
        """
        估计跟踪目标当前的速度值

        Parameters
        ----------
        box : 当前跟踪目标的位置
        pass_flag ： 当前跟踪目标的行进方向
        """
        if self.state != 0 and self.state != 1:  # 跟踪目标没有被删除，且跟踪目标未满足计算状态
            base_box = list(self.speed_queue.get())
            # 用来获取车辆跟踪点队列中的最早一个点，也就是速度估计的“起始点”。get()方法返回队列中的第一个元素,出队列（x1,y1,x2,y2）
            # 拿出来之后，默认就不再用了（也就是被“消费”了）；
            # 注意：这个操作是**“ destructive”** —— 会把数据从队列中移除掉
            pixel_dis = math.sqrt((box[2] - base_box[2]) ** 2 + (box[3] - base_box[3]) ** 2)  # 使用右下角框的像素变化值计算移动距离
            # distance, 像素距离
            if pass_flag == 1:  # 表示下行车辆
                self.box_speed = ((pixel_dis * self.down_speed_ref) / (0.1 * self.state)) * 3.6
                # 转换为实际距离速度
                # ( 两帧间的实际距离(米) / 用时(0.04s/帧) ) * 3.6(1m/s = 3.6km/h)
            else:
                self.box_speed = ((pixel_dis * self.up_speed_ref) / (0.1 * self.state)) * 3.6
        else:
            # 并没有用到所谓的3帧，5帧，10帧，里面的坐标
            self.box_speed = 45  # give an average speed of this road

        self.box_speed = round(self.box_speed, 2)
        # 作用：将速度四舍五入到小数点后 2 位。
        self.box_speed = min(self.box_speed, 200)  
        # 假定最高速度为200km/h

        # 销毁存储的速度队列，只保留速度与状态
        self.speed_queue.queue.clear()
        self.state = 1

        return self.box_speed

    def get_speed(self):
        """
        返回当前目标的速度
        """
        return self.box_speed

    def get_state(self):
        """
        返回当前目标速度估计的状态
        """
        return self.state


# 统计车辆数量、类别

########################################
# 一些参数的定义
# x是点到左边的距离，y是点到顶上的距离
# 小于则说明点落在直线与x轴所夹的锐角区域

# 方框顶点的序号
#    0 (x1,y1)      1 (x2, y1)
#    |--------------|
#    |              |
#    |              |
#    |--------------|
#    3 (x1,y2)      2 (x2, y2)


#    |-------> x轴
#    |
#    |
#    V
#    y轴

########################################
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

class ToolVehicle:
    """
        函数功能：主要是用来统计车辆数量
    """

    def __init__(self, line,FPS):
        # 车辆统计的相关参数
        self.small_to_big = 0  # 0表示从比线小的一侧往大的一侧， 表示车下行 （从上往下走）
        self.big_to_small = 1  # 1表示从检测线大的一侧往小的一侧   表示车上行
        self.down_point_idx = 2  # 要检测的方框顶点号(0, 1, 2, 3)，看下边的图，当方框的顶点顺着small_to_big指定的方向跨过检测线时，计数器会+1
        self.up_point_idx = 2

        self.total_num = 0  # 通过车辆的总计数
        self.up_last_frame_point = []  # 记录上行的追踪点
        self.down_last_frame_point = []  # 记录下行的追踪点
        self.up_has_pase_point = []  # 上行已经超过检测线的保存点
        self.down_has_pase_point = []  # 下行已经超过检测线的保存点
        self.down_xi, self.down_yi = self.point(self.down_point_idx)
        self.up_xi, self.up_yi = self.point(self.up_point_idx)

        # 计数区域初始化
        self.line = line  # 检测线（必须为一条纵坐标一致的横线）

        # 速度估计成员列表
        self.estimate_speed_list = {}
        # 速度估计成员列表
        self.FPS=FPS
        self.coordinates = LastUpdatedOrderedDict(maxlen=10000)
        self.car_class   = LastUpdatedOrderedDict(maxlen=10000)
        self.speed_end   = LastUpdatedOrderedDict(maxlen=10000)
        self.class_end   = LastUpdatedOrderedDict(maxlen=10000)
        self.up_car      = deque(maxlen=10000)
        self.down_car    = deque(maxlen=10000)
        self.count       = 0
        # 初始化Colors对象 下面调用colors的时候会调用__call__函数
        self.colors = Colors()

        # 数量存储字段
        self.classes = ['Motorcycle', 'Car', 'Bus', 'Tractor', 'L_truck', 'XL_truck', 'XXL_truck', 'XXXL_truck',
                        'Container car', 'Electric vehicle']

        self.down_count = {"Motorcycle": 0, "Car": 0, "Bus": 0, "Tractor": 0, "L_truck": 0, "XL_truck": 0,
                           "XXL_truck": 0,
                           "XXXL_truck": 0, "Container car": 0, "Electric vehicle": 0, "Total": 0}
        self.up_count = {"Motorcycle": 0, "Car": 0, "Bus": 0, "Tractor": 0, "L_truck": 0, "XL_truck": 0, "XXL_truck": 0,
                         "XXXL_truck": 0, "Container car": 0, "Electric vehicle": 0, "Total": 0}

    @staticmethod
    def point(point_idx):
        """
        转换两点坐标为特定坐标

        Parameters
        ----------
        point_idx : 选择特定坐标的位置
        """
        # x_i、y_i表示x、y在points数组中的下标  point(x1, y1, x2, y2)
        if point_idx == 0:
            x_i = 0  # 左上角的坐标
            y_i = 1
        elif point_idx == 1:
            x_i = 2  # 右上角坐标
            y_i = 1
        elif point_idx == 2:
            x_i = 2  # 右下角坐标
            y_i = 3
        elif point_idx == 3:
            x_i = 0  # 左下角坐标
            y_i = 3
        else:
            x_i = 2  # 右下角坐标
            y_i = 3
        return x_i, y_i

    def point_bigger(self, x, y) -> bool:
        """
        判断边界框的点是否超过检测线  表示从比线大的一侧往小的一侧  车辆从下到上

        Parameters
        ----------
        x : 当前判定点的x坐标
        y : 当前判定点的y坐标
        """
        x1 = self.line[0]
        y1 = self.line[1]
        x2 = self.line[2]
        y2 = self.line[3]

        if y1 == y2:    # 线左右两个点在同一条水平线上
            if y > y1:
                return True  # 检测边框的点 还没有 超过了检测线
            elif y <= y1:
                return False  # 检测边框的点 已经 超过了检测线

        if x1 == x2:
            if x > x1:
                return True
            elif x <= x1:
                return False

        if (x - x1) / (x2 - x1) > (y - y1) / (y2 - y1):
            return True
        else:
            return False

    def point_smaller(self, x, y) -> bool:
        """
        判断边界框的点是否超过检测线  表示从比线小的一侧往大的一侧  车辆从上到下

        Parameters
        ----------
        x : 当前判定点的x坐标
        y : 当前判定点的y坐标
        """
        x1 = self.line[0]
        y1 = self.line[1]
        x2 = self.line[2]
        y2 = self.line[3]

        if y1 == y2:
            if y < y1:  # 检测边框的点没有超过检测线
                return True
            elif y >= y1:
                return False  # 当检测边界框的点超过了检测线  计数+1

        if x1 == x2:
            if x < x1:
                return True
            elif x >= x1:
                return False

        if (x - x1) / (x2 - x1) < (y - y1) / (y2 - y1):
            return True
        else:
            return False

    def judge_size(self, direction, x, y):
        """
        判定车辆是否是向特定方向行进

        Parameters
        ----------
        direction : 车辆行进判定的方向
        x : 当前判定点的x坐标
        y : 当前判定点的y坐标
        """
        if direction == 0:  # 从小到大   车辆从上到下进行
            return self.point_smaller(x, y)
        elif direction == 1:  # 车辆从下到上
            return self.point_bigger(x, y)
        else:
            print('方向错误，只能为0或1！')

    def get_all_class_number(self, down_cls, up_cls):
        """
        计算检测图像中的预测出的各种车型数量

        Parameters
        ----------
        down_cls : 下行车辆的类别
        up_cls : 上行车辆的类别
        """
        if down_cls in self.classes:
            self.down_count[down_cls] += 1
            self.down_count["Total"] += 1
        if up_cls in self.classes:
            self.up_count[up_cls] += 1
            self.up_count["Total"] += 1

    def _initiate_estimate(self, track_id, point):
        """
        初始化速度估计成员

        Parameters
        ----------
        track_id : 车辆的跟踪编号
        point : 当前车辆的坐标
        """
        self.estimate_speed_list[str(track_id)] = EstimateSpeed(point)
    def track_count_and_estimate(self, point, track_id, name):
        """
        跟踪计数与速度估计

        Parameters
        ----------
        point : 当前车辆的坐标
        track_id : 车辆的跟踪编号
        name : 当前车辆的识别类别
        """
        # print("输出需要追踪的车ID",track_id)
        down_name = None
        up_name = None
        pass_flag = 0
        box_speed = 0

        # 如果存在当前id的速度估计成员，更新速度参数
        speed_update = self.estimate_speed_list.get(str(track_id))
        if speed_update is not None:
            speed_update.update(point)

        # 如果ID在上一帧，判断是否超过检测线
        if (track_id in self.down_last_frame_point) and (
                not self.judge_size(self.small_to_big, point[self.down_xi], point[self.down_yi])):
            # 如果ID在上一帧，判断是否超过检测线
            self.down_last_frame_point = [x for x in self.down_last_frame_point if x > track_id]  # 如果已经超过检测线，删除记录的ID
            print("输出下行每一帧追踪的车", self.down_last_frame_point)
            self.down_has_pase_point.append(track_id)
            print("输出下行已经保存", self.down_has_pase_point)
            self.total_num += 1
            down_name = name
            print("要删除的下行id1", track_id, name)
            pass_flag = 1
            # 计算估计速度值
            #box_speed = self.estimate_speed_list[str(track_id)].estimate_speed(point, pass_flag)
            box_speed = self.estimate_speed_list.get(str(track_id)).estimate_speed(point, pass_flag)
        elif (track_id in self.up_last_frame_point) and (
                not self.judge_size(self.big_to_small, point[self.up_xi], point[self.up_yi])):
            self.up_last_frame_point = [x for x in self.up_last_frame_point if x > track_id]  # 如果已经超过检测线，删除记录的ID
            print("输出上行每一帧追踪的车", self.up_last_frame_point)
            self.up_has_pase_point.append(track_id)
            print("输出已经保存的长度", self.up_has_pase_point)
            self.total_num += 1
            up_name = name
            print("要删除的上行id2", track_id, name)
            pass_flag = 2
            # 计算估计速度值
            #box_speed = self.estimate_speed_list[str(track_id)].estimate_speed(point, pass_flag)
            box_speed = self.estimate_speed_list.get(str(track_id)).estimate_speed(point, pass_flag)
        else:
            pass

        # 如果车辆的ID不在上一帧的记录点，并且边界框没有超过检测线，增加ID到上一帧记录点
        if ((track_id not in self.down_last_frame_point) and (track_id not in self.down_has_pase_point) and (
                track_id not in self.up_last_frame_point) and (track_id not in self.up_has_pase_point) and
                self.judge_size(self.small_to_big, point[self.down_xi], point[self.down_yi])):
            self.down_last_frame_point.append(track_id)
            self._initiate_estimate(track_id, point)  # 初始化速度估计成员
            print("增加的下行车的id", track_id, name)
        elif ((track_id not in self.up_last_frame_point) and (track_id not in self.up_has_pase_point) and (
                track_id not in self.down_last_frame_point) and (track_id not in self.down_has_pase_point) and
              self.judge_size(self.big_to_small, point[self.up_xi], point[self.up_yi])):
            self.up_last_frame_point.append(track_id)
            self._initiate_estimate(track_id, point)  # 初始化速度估计成员
            print("增加的上行车的id", track_id, name)
        else:
            pass

        # 如果存储列表过长，删除元素，防止长时间运行溢出
        if len(self.down_has_pase_point) > 1000:
            self.down_has_pase_point = [x for x in self.down_has_pase_point if x > (track_id - 100)]
        if len(self.up_has_pase_point) > 1000:
            self.up_has_pase_point = [x for x in self.up_has_pase_point if x > (track_id - 100)]
        if len(self.estimate_speed_list) > 1000:
            self.estimate_speed_list = {key: value for key, value in self.estimate_speed_list.items()
                                        if (int)(key) > (track_id - 100)}

        self.get_all_class_number(down_name, up_name)  # 统计上行和下行车的数量
        # 返回pass_flag, box_speed
        return pass_flag, box_speed
    
    from collections import Counter
 
    # def most_common_element(self,lst):
    #     # 使用Counter来统计每个元素的出现次数
    #     counts = Counter(lst)
    #     # 返回出现次数最多的元素
    #     return max(counts.elements(), key=counts.get)
    


    def counting(self, now_outputs, cls_names, offset=(0, 0)):
        """
        TODO：计数与测速实现

        Parameters
        ----------
        now_outputs : 当前图片帧检测跟踪输出值
        cls_names : 车辆的检测类别
        offset=(0, 0) : 坐标系的初始值
        """
        bbox_xyxy = now_outputs[:, :4]
        bbox_cls = now_outputs[:, -2]
        identities = now_outputs[:, -1]
        upload_info = []

        for i, box in enumerate(bbox_xyxy):
            if box[2] > 0 and box[3] > 0:  # bottom_right point > (0,0)
                x1, y1, x2, y2 = [int(i) for i in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]

                # 获取当前检测框的id与类别名
                track_id = int(identities[i]) if identities is not None else 0
                cls_name = cls_names[int(bbox_cls[i])]
                # 统计车辆的数量
                pass_flag, box_speed = self.track_count_and_estimate(box, track_id, cls_name)

                if pass_flag != 0:  # 0：还未经过统计线； 1：下行车辆； 2：上行车辆
                    upload_info.append([x1, y1, x2, y2, bbox_cls[i], box_speed, pass_flag])

        return self.total_num, self.down_count, self.up_count, upload_info

    def draw_boxes_speed(self, img, now_outputs, cls_names):
        """
        在图片中画出跟踪框与速度信息

        Parameters
        ----------
        img : 当前需要显示的图片
        now_outputs : 当前的跟踪输出内容
        cls_names : 检测的类别名
        """
        bbox_xyxy = now_outputs[:, :4]
        bbox_cls = now_outputs[:, -2]
        identities = now_outputs[:, -1]

        for i, box in enumerate(bbox_xyxy):
            # 进行一个车，一个检测框，一个ID进行计算相关功能

            if box[2] > 0 and box[3] > 0:  # bottom_right point > (0,0)
                x1, y1, x2, y2 = [int(i) for i in box]

                # box text and bar
                now_id = identities[i] if identities is not None else 0
                # 查询速度字典中是否存在对应id的速度，若查询失败则返回None
                box_speed = self.estimate_speed_list.get(str(int(now_id)))
                box_speed = box_speed.get_speed() if box_speed is not None else '?'
                name = cls_names[int(bbox_cls[i])]
                color = self.colors(bbox_cls[i])

                label = str(now_id) + ':' + str(name) + ":" + str(box_speed) + 'km/h'  # 显示车辆的id + 速度
                # label = '{} {:.2f}km/h'.format(str(name), speed)
                # label = '{}'.format(str(name))  # 显示车辆类别
                # label = '{}{}'.format(id, str(name)) # 显示车辆的id + 类别
                # label = '{}{:.2f}'.format(str(name), conf[i])  # 显示车辆的类别 + 置信度

                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # 画出车型的预测框
                cv2.line(img,(self.line[0],self.line[1]),(self.line[2],self.line[3]),color=[0,255,0])
                # cv2.circle(img, (x2, y2), radius=4, color=(0, 0, 255), thickness=5)  # 将预测框右下角标出来
                cv2.rectangle(
                    img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)  # 画出标签的背景框
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # 写出标签

    def draw_count_res(self, img):
        """
        显示统计车辆的内容

        Parameters
        ----------
        img : 当前需要显示的图片
        """

        # 显示统计车辆线 画布、起点坐标、终点坐标、线颜色、线粗细
        cv2.line(img, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 255, 0), 2)
        cv2.putText(img, f'{str("Total Vehicle Flow"):<10} = {self.total_num:>2}', (25, 35),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (130, 210, 255), 2)  # 画布、内容、左下角坐标、字体、字号（数字大字跟着大）、字颜色、笔画粗细
        cv2.putText(img, f'{str("Category"):<17}{str("DOWN"):>8}{str("UP"):>7}', (25, 70), cv2.FONT_HERSHEY_COMPLEX,
                    0.75,
                    (255, 255, 255), 2)  # 画布、内容、左下角坐标、字体、字号（数字大字跟着大）、字颜色、笔画粗细
        cv2.putText(img, f'{str("Total"):<18}', (25, 95), cv2.FONT_HERSHEY_COMPLEX,
                    0.75, (255, 255, 255), 2)  # 画布、内容、左下角坐标、字体、字号（数字大字跟着大）、字颜色、笔画粗细
        cv2.putText(img, f'{self.down_count["Total"]:>7}{self.up_count["Total"]:>9}', (250, 95),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.75, (255, 255, 255), 2)  # 画布、内容、左下角坐标、字体、字号（数字大字跟着大）、字颜色、笔画粗细

        for i, cls in enumerate(self.classes):
            cv2.putText(img, f'{str(cls):<18}', (25, 120 + 25 * i), cv2.FONT_HERSHEY_COMPLEX, 00.75, (255, 255, 255),
                        2)  # 画布、内容、左下角坐标、字体、字号（数字大字跟着大）、字颜色、笔画粗细

            cv2.putText(img, f'{self.down_count[cls]:>7}{self.up_count[cls]:>9}', (250, 120 + 25 * i),
                        cv2.FONT_HERSHEY_COMPLEX, 00.75, (255, 255, 255),
                        2)  # 画布、内容、左下角坐标、字体、字号（数字大字跟着大）、字颜色、笔画粗细
