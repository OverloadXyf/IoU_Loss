 ## IoU Loss 说明 非官方实现 Not Official !!!
 ### IOU GIoU DIOU CIOU EIOU 
 #### *IOU LOSS=1-IoU*
 #### *GIOU LOSS=1-GIoU*
 #### *DIOU LOSS=1-DIoU*
 #### *CIOU LOSS=1-CIoU*
 #### Focal EIoU Loss
 定义为：$ IOU^{\gamma } \times EIOU $

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
    print(res)
```

 ### References

 #### GIOU 2019
    -Paper: Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression
    -https://arxiv.org/pdf/1902.09630

#### DIOU & CIOU 2019
    -Paper: Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression 
    -https://arxiv.org/abs/1911.08287

#### EIOU 2021
    -Paper: Focal and Efficient IOU Loss for Accurate Bounding Box Regression。
    -https://arxiv.org/abs/2101.08158

#### SIOU 2022
    -Paper: SIoU Loss: More Powerful Learning for Bounding Box Regression
    -https://arxiv.org/abs/2205.12740


