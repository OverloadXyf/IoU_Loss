import torch
import math
from IoUs import generalized_box_iou, distant_iou, box_iou

'''
    联合IoU：
    纵观IoU的讨论：
        1）Giou 在意能否将真值框填满 从面积上讨论
        2）Diou 在意两个框的中心点距离是否相近
    
    我们发现 两个框即使中心点相近 但也可能面积不匹配
    因此第一种联合IoU的方式便是 联合两个IOU
    
    Loss_GDIoU=1-GDIoU

'''


def Union_GDIoU(boxes1, boxes2):
    # 保证数据的正确性做的事先判断
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, _ = box_iou(boxes1, boxes2)

    # Giou_term = - (area - union) / area
    Giou_term = generalized_box_iou(boxes1, boxes2) - iou

    # Diou_term = - 1.0 * square_center / diag_line
    Diou_term = distant_iou(boxes1, boxes2) - iou

    # GDiou = iou-(area - union) / area- 1.0 * square_center / diag_line
    GD_iou = iou + Giou_term + Diou_term

    return GD_iou


'''
    联合Focal loss 和 GIoU 的方法
    类比Focal loss 与 EIoU 的结合

'''


def Focal_GIoU_loss(boxes1, boxes2, gamma=0.5):
    # 保证数据的正确性做的事先判断
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, _ = box_iou(boxes1, boxes2)

    GIoU = generalized_box_iou(boxes1, boxes2)

    focal_giou_loss = torch.pow(iou, gamma) * (1-GIoU)

    return focal_giou_loss


if __name__ == '__main__':
    print(math.sqrt(0.1))
