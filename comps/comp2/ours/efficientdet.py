import numpy as np
import colorsys
import pickle
import os
import tensorflow as tf
from model import Efficientdet
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from PIL import Image,ImageFont, ImageDraw
from utils.utils import BBoxUtility,letterbox_image,efficientdet_correct_boxes
from utils.anchors import get_anchors

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
def preprocess_input(image):
    image /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    return image

class EfficientDet(object):
    _defaults = {
        "model_path"    : 'logs/phi2-ep043-loss0.424-val_loss0.443.h5',
        "classes_path"  : 'model_data/voc_classes.txt',
        "phi"           : 2,
        "confidence"    : 0.25,
        "iou"           : 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.model_image_size = [image_sizes[self.phi], image_sizes[self.phi],3]
        self.generate()
        self.bbox_util = BBoxUtility(self.num_classes,nms_thresh=self.iou)
        self.prior = self._get_prior()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_prior(self):
        data = get_anchors(image_sizes[self.phi])
        return data

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 計算 class 種類數
        self.num_classes = len(self.class_names)

        # 載入模型
        self.Efficientdet = Efficientdet(self.phi,self.num_classes)
        self.Efficientdet.load_weights(self.model_path,by_name=True,skip_mismatch=True)

        # self.Efficientdet.summary()
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 畫框設置不同的顏色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = letterbox_image(image, [self.model_image_size[0],self.model_image_size[1]])
        photo = np.array(crop_img,dtype = np.float32)

        # prepocess image
        photo = np.reshape(preprocess_input(photo),[1,self.model_image_size[0],self.model_image_size[1],self.model_image_size[2]])

        preds = self.Efficientdet.predict(photo)
        # decode prediction results
        results = self.bbox_util.detection_out(preds, self.prior, confidence_threshold=self.confidence)
        if len(results[0])<=0:
            return image
        results = np.array(results)

        # 篩選出其中得分高於 confidence 的框
        det_label = results[0][:, 5]
        det_conf = results[0][:, 4]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]
        
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 畫框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 抖動，讓線條比較粗一點
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def output_txt(self, image):
        #print("debug")
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = letterbox_image(image, [self.model_image_size[0],self.model_image_size[1]])
        photo = np.array(crop_img,dtype = np.float32)
        #print("debug1")
        photo = np.reshape(preprocess_input(photo),[1,self.model_image_size[0],self.model_image_size[1],self.model_image_size[2]])

        preds = self.Efficientdet.predict(photo)
        #print("debug2")
        results = self.bbox_util.detection_out(preds, self.prior, confidence_threshold=self.confidence)
        if len(results[0])<=0:
            return [], [], [], [], [], []
        results = np.array(results)
        #print("debug3 results: ", results)

        det_label = results[0][:, 5]
        det_conf = results[0][:, 4]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]
        
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        #print("debug4: ", top_xmin, top_ymin, top_xmax, top_ymax)
        boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
        
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for i, c in enumerate(top_label_indices):

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
            
            x0.append(top)
            y0.append(left)
            x1.append(bottom)
            y1.append(right)
        #print(x0, y0, x1, y1, top_label_indices, top_conf)
        return x0, y0, x1, y1, top_label_indices, top_conf