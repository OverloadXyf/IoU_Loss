 ##IoU Loss 说明
 ### IOU GIoU DIOU CIOU EIOU 
 #### *IOU LOSS=1-IoU*
 #### *GIOU LOSS=1-GIoU*
 #### *DIOU LOSS=1-DIoU*
 #### *CIOU LOSS=1-CIoU*
 $ IOU^{\gamma } \times EIOU $

 #### 可以使用以下简单代码快验证定义的iou是否计算正确
 #### 相对来说 EIOU比较小
```python
import torch
from IoUs import efficient_iou

if __name__ == '__main__':
    a = torch.tensor([[0, 0, 2, 2]])
    b = torch.tensor([[0, 0, 3, 3],
                      [1, 1, 5, 5]])
    res = efficient_iou(a, b)
    # res =generalized_box_iou(a,b)
```


