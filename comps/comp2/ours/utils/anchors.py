import numpy as np
from tensorflow import keras
import tensorflow as tf

class AnchorParameters:
    # ratios: 長寬比-> 0.5代表 w:h = 1:2
    # scales: 針對不同倍數放大 anchors
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
)

def generate_anchors(base_size=16, ratios=None, scales=None):
    ####################################
    # 生成 9 個 anchors
    ####################################
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))
    #print("anchors1: ", anchors)

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T # .T 是做轉置
    #[[ 0.          0.         16.         16.        ]
    # [ 0.          0.         20.15873718 20.15873718]
    # [ 0.          0.         25.39841652 25.39841652]
    # [ 0.          0.         16.         16.        ]
    # [ 0.          0.         20.15873718 20.15873718]
    # [ 0.          0.         25.39841652 25.39841652]
    # [ 0.          0.         16.         16.        ]
    # [ 0.          0.         20.15873718 20.15873718]
    # [ 0.          0.         25.39841652 25.39841652]]
    #print("anchors1: ", anchors)

    areas = anchors[:, 2] * anchors[:, 3]
#     [256.         406.3746848  645.07956168 256.         406.3746848
#  645.07956168 256.         406.3746848  645.07956168]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    #print("anchors:\n", anchors)
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
# [[ 0.          0.         22.627417   11.3137085 ]
#  [ 0.          0.         28.50875952 14.25437976]
#  [ 0.          0.         35.9187851  17.95939255]
#  [ 0.          0.         16.         16.        ]
#  [ 0.          0.         20.15873718 20.15873718]
#  [ 0.          0.         25.39841652 25.39841652]
#  [ 0.          0.         11.3137085  22.627417  ]
#  [ 0.          0.         14.25437976 28.50875952]
#  [ 0.          0.         17.95939255 35.9187851 ]]
    #print("anchors:\n", anchors)

    #print("anchors:\n", anchors[:, 0::2])
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
# [[-11.3137085    0.          11.3137085   11.3137085 ]
#  [-14.25437976   0.          14.25437976  14.25437976]
#  [-17.95939255   0.          17.95939255  17.95939255]
#  [ -8.           0.           8.          16.        ]
#  [-10.07936859   0.          10.07936859  20.15873718]
#  [-12.69920826   0.          12.69920826  25.39841652]
#  [ -5.65685425   0.           5.65685425  22.627417  ]
#  [ -7.12718988   0.           7.12718988  28.50875952]
#  [ -8.97969628   0.           8.97969628  35.9187851 ]]
    #print("anchors:\n", anchors)
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
# [[-11.3137085   -5.65685425  11.3137085    5.65685425]
#  [-14.25437976  -7.12718988  14.25437976   7.12718988]
#  [-17.95939255  -8.97969628  17.95939255   8.97969628]
#  [ -8.          -8.           8.           8.        ]
#  [-10.07936859 -10.07936859  10.07936859  10.07936859]
#  [-12.69920826 -12.69920826  12.69920826  12.69920826]
#  [ -5.65685425 -11.3137085    5.65685425  11.3137085 ]
#  [ -7.12718988 -14.25437976   7.12718988  14.25437976]
#  [ -8.97969628 -17.95939255   8.97969628  17.95939255]]
    #print("anchors:\n", anchors)

    return anchors

def shift(shape, stride, anchors):
    ####################################
    # 將不同大小的 anchors 安置在圖形裡面
    ####################################
    shift_x = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    # print("shift_x:\n", shift_x)
    # print("shift_y:\n", shift_y)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    # 存成 anchors 的兩個座標的位移量
    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    # 位移 anchors 並存成下面格式
    # [
    #     [x1, y1, x2, y2]
    #     ...
    # ]
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    # print(np.shape(shifted_anchors))
    # print(shifted_anchors)
    return shifted_anchors

def get_anchors(image_size):
    border = image_size # 根據phi有不同的image_size
    features = [image_size/8,image_size/16,image_size/32,image_size/64,image_size/128]
    shapes = []
    for feature in features:
        shapes.append(feature)
    all_anchors = []
    for i in range(5):
        anchors = generate_anchors(AnchorParameters.default.sizes[i])
        shifted_anchors = shift([shapes[i],shapes[i]], AnchorParameters.default.strides[i], anchors)
        all_anchors.append(shifted_anchors)

    # np.concatenate 會將不同大小的 anchors 
    # [
    #     [x1, y1, x2, y2]
    #     ...
    # ]
    # 拼接起來
    all_anchors = np.concatenate(all_anchors,axis=0)
    all_anchors = all_anchors/border
    # all_anchors = all_anchors.clip(0,1)
    # print(np.shape(all_anchors))
    return all_anchors