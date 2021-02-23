import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import math


def letterbox_image(image, size):
    # 生出正確 size 的圖片
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def efficientdet_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    # 存箱子的中心點和高跟寬
    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    # 調整箱子位置跟大小
    box_yx = (box_yx - offset) * scale
    box_hw *= scale


    box_mins = box_yx - (box_hw / 2.) # 箱子左下角的點座標
    box_maxes = box_yx + (box_hw / 2.) # 箱子右上角的點座標
    # 把箱子的左下右上存起來
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

class PriorProbability(keras.initializers.Initializer): # 初始化一開始的權重
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape) * -math.log((1 - self.probability) / self.probability)

        return result

class BBoxUtility(object):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,ignore_threshold=0.4,
                 nms_thresh=0.3, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
    def iou(self, box):
        # inter_upleft: (priors box) 和 (預測 box) 交集的左上座標
        # inter_botright: (priors box) 和 (預測 box) 交集的右下座標
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        # 算 (priors box) 和 (預測 box) 交集的面積
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 預測 box 的面積
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # priors box 的面積
        area_gt = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])
        
        # 計算 IOU
        union = area_true + area_gt - inter
        iou = inter / union
        
        return iou

    # 對 box encode ，就是把 gt box 的座標變成 feature map 的
    # 有頂點座標 -> (x-center, y-center, w, h) -> 在轉換成對應 feature map 下的尺寸
    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到與 box IOU 較高的 priors box，全部都抓出來
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        # 找到對應的 priors box
        assigned_priors = self.priors[assign_mask]

        #########################################
        # 將 box 轉化為 efficientdet 預測結果的格式
        #########################################
        
        # box 的中心和長寬
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]

        # 計算重合度較高的 priors 中心和長寬
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        
        # 逆向求取 efficientdet 應該有的預測結果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        return encoded_box.ravel()
    def ignore_box(self, box):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))

        assign_mask = (iou > self.ignore_threshold)&(iou<self.overlap_threshold)

        # 如果全部的 iou 都 小於0.4 或者 大於0.5，那就挑一個最大的設成 True
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
            
        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.
        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.
        """
        assignment = np.zeros((self.num_priors, 4 + 1 + self.num_classes + 1))
        assignment[:, 4] = 0.0
        assignment[:, -1] = 0.0
        if len(boxes) == 0:
            return assignment

        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])

        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        # (num_priors)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        # (num_priors)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, 4][ignore_iou_mask] = -1
        assignment[:, -1][ignore_iou_mask] = -1

        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # box encoded 後的值和iou
        # (n, num_priors, 5)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合程度最大的 priors，並且獲得 idx
        # (num_priors)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # (num_priors)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 此 priors 屬於哪個 box
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的 priors box 的預測結果
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
        # 4 代表一定不會被當成背景
        assignment[:, 4][best_iou_mask] = 1
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -1][best_iou_mask] = 1
        # 通過 assign_boxes 可以得到 input_image 應該有的預測結果是怎麼樣的

        return assignment
        
    def decode_boxes(self, mbox_loc, mbox_priorbox):
        '''
        反求出預測的座標，在原圖中的位置
        
        由 (center_x, center_y, w, h) -> (xmin, ymin, xmax, ymax)
        '''
        # prior box 的 w, h, center_x, center_y
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # box 距離 prior box center xy axis 偏移情況
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height
        decode_bbox_center_y += prior_center_y
        
        # box 的 w, h
        decode_bbox_width = np.exp(mbox_loc[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3])
        decode_bbox_height *= prior_height

        # box 的左上和右下座標
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # box 的左上右下座標合併起來 存成一個 list
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)

        # 防止超出0, 1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, mbox_priorbox, confidence_threshold=0.4):
        # network 預測的結果
        mbox_loc = predictions[0]
        # confidence
        mbox_conf = predictions[1]
        # prior box
        mbox_priorbox = mbox_priorbox
        
        results = []
        # 對每個圖片進行處理
        for i in range(len(mbox_loc)):
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)

            bs_class_conf = mbox_conf[i]
            
            class_conf = np.expand_dims(np.max(bs_class_conf, 1),-1)
            class_pred = np.expand_dims(np.argmax(bs_class_conf, 1),-1)

            conf_mask = (class_conf >= confidence_threshold)[:,0]

            detections = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]), 1)
            # 看看這張圖片有哪些不同的 class
            unique_class = np.unique(detections[:,-1])

            best_box = []
            if len(unique_class) == 0:
                results.append(best_box)
                continue
            # 對 class 進行循環，篩選出一定區域內屬於同一種類得分最大的框，
            # 對 class 進行循環可以幫助我們對每一個 class 分別進行 非極大抑制
            for c in unique_class:
                cls_mask = detections[:,-1] == c

                detection = detections[cls_mask]
                scores = detection[:,4]
                # 根據得分將此 class 由大到小 sort
                arg_sort = np.argsort(scores)[::-1]
                detection = detection[arg_sort]
                while np.shape(detection)[0]>0:
                    # 每次取出得分最大的 box 計算他跟其他 prior box 的重合程度，
                    # 重合成度過大的剔掉，因為我們不想要框到重複的東西
                    best_box.append(detection[0])
                    if len(detection) == 1:
                        break
                    ious = iou(best_box[-1],detection[1:])
                    detection = detection[1:][ious<self._nms_thresh]
            results.append(best_box)
        # results 存所有預測結果裡面 confidence 比較高的 box
        return results

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

