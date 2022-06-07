import torch
import math

"""
torch.unbind(x,dim) 在指定的dim上做切片
torch.cat 是直接在指定的维度拼接 不改变输入的维度 输入是2维 输出也是2维
torch.stack 是在指定增加一个维度上再堆叠
"""


def box_area(boxes: torch.Tensor):
    """
    计算面积 这里的输入需要是 (x1,y1,x2,y2)的格式 其中后面的坐标大于前面的坐标
    这里可以使用 from torchvision.ops.boxes import box_area

    :param boxes: (x1,y1,x2,y2)的格式 为[N,4]大小
    :return: tensor.Tensor[N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # N 表示为(x2-x1)*(y2-y1)


def box_cxcywh_xyxy(x):
    # x.shape= N 4 --->x_c.shape= N
    x_c, y_c, w, h = x.unbind(-1)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=-1)  # N 在最后增加1维后堆叠 变成 N 4


def box_xyxy_to_cxcywh(x):
    # x.shape= N 4 --->x_c.shape= N
    x0, y0, x1, y1 = x.unbind(-1)

    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]

    return torch.stack(b, dim=-1)  # N 在最后增加1维后堆叠 变成 N 4


