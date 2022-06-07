import torch
import math
from box_convert import box_area
# from torchvision.ops.boxes import box_area



def box_iou(boxes1, boxes2):
    """
    这里返回iou和union 方便别的IOU的计算
    :param boxes1: shape= N 4
    :param boxes2: shape= M 4
    :return:iou shape= N M 每个框对应的IOU;union shape= N M 每个框对应的union
    """
    area1 = box_area(boxes1)  # N
    area2 = box_area(boxes2)  # M

    # Intersection Area
    # 1.两者较小的坐标中 较大的一个
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # N M 2

    # 2.两者较大的坐标中 较小的一个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # N M 2

    # 3 得到宽高信息 用大的减去小的 clamp保证大于等于0
    wh = (rb - lt).clamp(min=0)  # N M 2

    # 4 Inter面积
    inter = wh[:, :, 0] * wh[:, :, 1]  # N M

    union = area1[:, None] + area2 - inter  # N M 计算出NM个框的union 所以要用到None

    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    '''

    :param boxes1: shape= N 4
    :param boxes2: shape= M 4
    :return: giou shape= N M
    '''

    # 保证数据的正确性做的事先判断
    assert (boxes1[:, :2] <= boxes1[:, 2:]).all()
    assert (boxes2[:, :2] <= boxes2[:, 2:]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 找最小包围框的坐标位置
    # 1.较大坐标的最大坐标
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    # 2.较小坐标中的最小坐标
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])

    # wh_minbox
    wh_minbox = (rb - lt).clamp(min=0)  # N M 2
    area = wh_minbox[:, :, 0] * wh_minbox[:, :, 1]  # N M

    return iou - (area - union) / area


def distant_iou(boxes1, boxes2):
    '''
    :param boxes1: shape= N 4
    :param boxes2: shape= M 4
    :return: diou shape= N M
    '''

    # 保证数据的正确性做的事先判断
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 大框的左上角
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # 大框的右下角
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 相减得大于0
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    # 包围矩形的对角线的平方
    diag_line = torch.pow(wh[:, :, 0], 2) + torch.pow(wh[:, :, 1], 2)  # [N,M]

    # 两矩形中心点的平方
    # 矩形中心点
    boxes1_center_x = torch.true_divide(boxes1[..., 0] + boxes1[..., 2], 2)
    boxes1_center_y = torch.true_divide(boxes1[..., 1] + boxes1[..., 3], 2)  # [N]

    boxes2_center_x = torch.true_divide(boxes2[:, 0] + boxes2[:, 2], 2)
    boxes2_center_y = torch.true_divide(boxes2[:, 1] + boxes2[:, 3], 2)  # [M]

    dis_cx = torch.pow(boxes1_center_x.unsqueeze(-1)[:, None, 0] - boxes2_center_x.unsqueeze(-1)[:, 0], 2)  # [N,M]
    dis_cy = torch.pow(boxes1_center_y.unsqueeze(-1)[:, None, 0] - boxes2_center_y.unsqueeze(-1)[:, 0], 2)  # [N,M]

    square_center = dis_cx + dis_cy  # [N,M]

    return iou - 1.0 * square_center / diag_line


def complete_iou(boxes1, boxes2):
    '''
    :param boxes1: shape= N 4
    :param boxes2: shape= M 4
    :return: ciou shape= N M
    '''

    # 保证数据的正确性做的事先判断
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 大框的左上角
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # 大框的右下角
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 相减得大于0
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    # 包围矩形的对角线的平方
    diag_line = torch.pow(wh[:, :, 0], 2) + torch.pow(wh[:, :, 1], 2)

    # 两矩形中心点的平方
    # 矩形中心点
    boxes1_center_x = torch.true_divide(boxes1[..., 0] + boxes1[..., 2], 2)
    boxes1_center_y = torch.true_divide(boxes1[..., 1] + boxes1[..., 3], 2)  # [N]

    boxes2_center_x = torch.true_divide(boxes2[:, 0] + boxes2[:, 2], 2)
    boxes2_center_y = torch.true_divide(boxes2[:, 1] + boxes2[:, 3], 2)  # [M]

    dis_cx = torch.pow(boxes1_center_x.unsqueeze(-1)[:, None, 0] - boxes2_center_x.unsqueeze(-1)[:, 0], 2)  # [N,M]
    dis_cy = torch.pow(boxes1_center_y.unsqueeze(-1)[:, None, 0] - boxes2_center_y.unsqueeze(-1)[:, 0], 2)  # [N,M]

    square_center = dis_cx + dis_cy  # [N,M]

    # 得到宽高
    boxes1_w = boxes1[:, 2] - boxes1[:, 0]  # [N]
    boxes1_h = boxes1[:, 3] - boxes1[:, 1]
    boxes2_w = boxes2[:, 2] - boxes2[:, 0]
    boxes2_h = boxes2[:, 3] - boxes2[:, 1]  # [M]

    # atan只是减法的顺序 这里 boxes2才是真值框
    atan1 = torch.atan(boxes2_w / (boxes2_h + 1e-9))  # [M]
    atan2 = torch.atan(boxes1_w / (boxes1_h + 1e-9))  # [N]

    v = 4.0 * torch.pow((atan1 - atan2[:, None]), 2) / (math.pi ** 2)  # [N,M]
    # print('v',v)
    # print('v',v.shape)
    # print('iou',iou.shape)
    a = (v) / (1 + (1e-5) - iou + v)
    # print(a)
    ciou = iou - 1.0 * square_center / diag_line - 1.0 * a * v

    return ciou


if __name__ == '__main__':
    a = torch.rand(1, 4)
    print(a)
    print(a[..., 0])
