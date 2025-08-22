import numpy as np
import matplotlib.pyplot as plt

# 假设已有 im0 是 (C, H, W) 格式的 ndarray
# 示例：生成随机图像
im0 = np.random.randint(0, 256, (3, 384, 640), dtype=np.uint8)

# 框坐标（左上角和右下角）
x1, y1, x2, y2 = 100, 50, 300, 200

# 1️⃣ CHW → HWC 格式转换（matplotlib 要求 HWC 格式）
im_hwc = np.transpose(im0, (1, 2, 0))  # (C, H, W) → (H, W, C)

# 2️⃣ 创建绘图窗口
fig, ax = plt.subplots()
ax.imshow(im_hwc)

# 3️⃣ 添加矩形框（绿色框，线宽 2）
rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     edgecolor='lime', facecolor='none', linewidth=2)
ax.add_patch(rect)

# 4️⃣ 去除坐标轴、展示图像
ax.axis('off')
plt.title("Image with Bounding Box")
plt.show()
