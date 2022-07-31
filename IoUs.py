import torch
import math
from box_convert import box_area, box_xyxy_to_cxcywh


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
    print('inter=', inter)
    print('union=', union)

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


def efficient_iou(boxes1, boxes2):
    '''
        :param boxes1: shape= N 4
        :param boxes2: shape= M 4
        :return: eiou shape= N M
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
    diag_line = torch.pow(wh[:, :, 0], 2) + torch.pow(wh[:, :, 1], 2)  # N,M

    # 中心点位置距离的平方
    c_boxes1 = box_xyxy_to_cxcywh(boxes1)
    c_boxes2 = box_xyxy_to_cxcywh(boxes2)
    c_wh = torch.abs(c_boxes1[:, None, :2] - c_boxes2[:, :2])  # N,M,2
    center_dis = torch.pow(c_wh[:, :, 0], 2) + torch.pow(c_wh[:, :, 1], 2)  # N,M
    loss_dis = torch.true_divide(center_dis, diag_line)

    # 宽的距离之差的平方
    dis_width = torch.pow((c_boxes1[:, None, 2] - c_boxes2[:, 2]), 2)  # N,M
    # 最小包围框的width的平方
    enclose_width = torch.pow(wh[:, :, 0], 2)  # N,M
    loss_asp1 = torch.true_divide(dis_width, enclose_width)

    # 高的距离之差的平方
    dis_height = torch.pow((c_boxes1[:, None, -1] - c_boxes2[:, -1]), 2)  # N,M
    # 最小包围框的height的平方
    enclose_height = torch.pow(wh[:, :, 1], 2)  # N,M
    loss_asp2 = torch.true_divide(dis_height, enclose_height)

    eiou = iou - loss_dis - loss_asp1 - loss_asp2

    return eiou


def focal_efficient_iou_loss(boxes1, boxes2, gamma=0.5):
    '''
        :param boxes1: shape= N 4
        :param boxes2: shape= M 4
        :return: eiou shape= N M
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
    diag_line = torch.pow(wh[:, :, 0], 2) + torch.pow(wh[:, :, 1], 2)  # N,M

    # 中心点位置距离的平方
    c_boxes1 = box_xyxy_to_cxcywh(boxes1)
    c_boxes2 = box_xyxy_to_cxcywh(boxes2)
    c_wh = torch.abs(c_boxes1[:, None, :2] - c_boxes2[:, :2])  # N,M,2
    center_dis = torch.pow(c_wh[:, :, 0], 2) + torch.pow(c_wh[:, :, 1], 2)  # N,M
    loss_dis = torch.true_divide(center_dis, diag_line)

    # 宽的距离之差的平方
    dis_width = torch.pow((c_boxes1[:, None, 2] - c_boxes2[:, 2]), 2)  # N,M
    # 最小包围框的width的平方
    enclose_width = torch.pow(wh[:, :, 0], 2)  # N,M
    loss_asp1 = torch.true_divide(dis_width, enclose_width)

    # 高的距离之差的平方
    dis_height = torch.pow((c_boxes1[:, None, -1] - c_boxes2[:, -1]), 2)  # N,M
    # 最小包围框的height的平方
    enclose_height = torch.pow(wh[:, :, 1], 2)  # N,M
    loss_asp2 = torch.true_divide(dis_height, enclose_height)

    eiou = iou - loss_dis - loss_asp1 - loss_asp2  # N,M

    focal_eiou_loss = torch.pow(iou, gamma) * (1 - eiou)  # N,M

    return focal_eiou_loss


def SCYLLA_IoU(boxes1, boxes2):
    '''
    SIoU:https://arxiv.org/abs/2205.12740
    :param boxes1: boxes1: shape= N 4
    :param boxes2: boxes2: shape= M 4
    :return: SIoU: shape= N M
    '''

    # 保证数据的正确性做的事先判断
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)  # [M]
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)  # [M]
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)  # [N]
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)  # [N]

    # 大框的左上角
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [M,N,2]
    # 大框的右下角
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [M,N,2]

    # 大框的宽度
    cw = rb[:, :, 0] - lt[:, :, 0]  # [M,N]
    # 大框的高度
    ch = rb[:, :, 1] - lt[:, :, 1]  # [M,N]

    # SIOU中对两个框的宽度和高度的判断公司还得看论文
    s_cw = (boxes2[:, 0] + boxes2[:, 2] - boxes1[:, None, 0] - boxes1[:, None, 2]) * 0.5  # [M,N]
    s_ch = (boxes2[:, 1] + boxes2[:, 3] - boxes1[:, None, 1] - boxes1[:, None, 3]) * 0.5  # [M,N]

    # 将s_ch和s_cw作为三角形的直角边 sigma作为斜边
    sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)  # [M,N]

    # 求三角形中两个锐角的sin
    sin_alpha_1 = torch.abs(s_cw) / sigma  # [M,N]
    sin_alpha_2 = torch.abs(s_ch) / sigma  # [M,N]

    # 定义一个45度角的sin阈值
    threshold = pow(2, 0.5) / 2
    # 需要用的角是小于45度的 要选出alpha1 alpha2小于45度的角
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)  # [M,N]

    # 角度的代价值 因为是cos函数 值域为[0,1] 45度角的代价最大为1 0度的角代价最小为0
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)  # [M.N]

    # 之后考虑距离的代价 先求得宽高差异 这里与最小包围框比较
    rho_x = (s_cw / cw) ** 2  # [M.N]
    rho_y = (s_ch / ch) ** 2  # [M.N]
    # 这里的gamma 的值域在[-2,-1] 与角度相关 45度为-1 0度角为-2
    gamma = angle_cost - 2
    # 这里的distance cost 这个与角度也有关系
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)  # [M.N]

    # 接下来考虑的是shape cost
    omiga_w = torch.abs(w1[:,None] - w2) / torch.max(w1, w2) # [M,N]
    omiga_h = torch.abs(h1[:,None] - h2) / torch.max(h1, h2) # [M,N]
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4) # [M,N]

    # iou
    siou=iou-0.5*(distance_cost+shape_cost)

    return siou




if __name__ == '__main__':
    a = torch.tensor([[0, 0, 2, 2]])
    b = torch.tensor([[1, 1, 3, 3],
                      [1, 0, 3, 2]])
    res=SCYLLA_IoU(a, b)
    print(1-res)
    # res =generalized_box_iou(a,b)
