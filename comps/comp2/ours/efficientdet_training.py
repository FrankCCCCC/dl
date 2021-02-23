from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
from random import shuffle
from utils import backend
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

def preprocess_input(image):
    image /= 255
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Generator(object):

    def __init__(self, bbox_util,batch_size,train_lines, val_lines, image_size,num_classes):
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.train_batches = len(train_lines)
        self.val_batches = len(val_lines)
        self.image_size = image_size
        self.num_classes = num_classes

    #對圖片做隨機的變化處理
    def get_random_data(self, annotation_line, input_shape):

        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        #圖片中的標註框box,包含4個邊界點和1個類別；
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        #對圖片隨機轉成nh,nw的新尺寸圖片
        new_ar = w/h * rand(0.5,1.5)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #在新尺寸圖片上,用灰色方塊填滿空格
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #隨機左右翻轉圖片
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #先將圖片轉至HSV座標域中,隨機改變圖片的顏色範圍,再轉回RGB座標域
        hue = rand(-36, 36)
        sat = rand(0.66, 1.5)
        val = rand(0.66, 1.5) 
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue 
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        #完成圖片變換後,用box檢測變換是否有超出範圍
        #返回變化過後的圖片和 box_data其中包含 (xmin, ymin, xmax, ymax)和一個類別
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        
        #檢查box_data查看圖片是否符合邊界,不符合回傳空box
        if len(box) == 0:
            return image_data, []
        elif(box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    #將變化過的圖片和未變化圖片的預測結果結合,並回傳
    def generate(self, train = True, eager = True):
        while True:
            #打亂圖片順序
            if train:
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
            inputs = []
            target0 = []
            target1 = []
            n = len(lines)

            
            for i in range(len(lines)):
                #對圖片進行變化並計算box值
                img,y = self.get_random_data(lines[i], self.image_size[0:2])
                i = (i+1) % n
                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]
                    one_hot_label = np.eye(self.num_classes)[np.array(y[:,4],np.int32)]
                    #略過邊界異常的圖片
                    if ((boxes[:, 3]-boxes[:, 1]) <= 0).any() and ((boxes[:, 2]-boxes[:, 0])<=0).any():
                        continue 
                    y = np.concatenate([boxes,one_hot_label],axis=-1)

                    #預測未變化圖片的輸出結果
                    assignment = self.bbox_util.assign_boxes(y)
                    regression = assignment[:,:5]
                    classifier = assignment[:,5:]

                    #將變化過的圖片當作input,將未變化圖片的預測結果當作變化後圖片的輸出
                    inputs.append(preprocess_input(img))         
                    target0.append(np.reshape(regression,[-1,5]))
                    target1.append(np.reshape(classifier,[-1,self.num_classes+1]))
                    if len(target0) == self.batch_size:
                        tmp_inp = np.array(inputs)
                        tmp_targets = [np.array(target0,dtype=np.float32),np.array(target1,dtype=np.float32)]
                        inputs = []
                        target0 = []
                        target1 = []
                        if eager:
                            yield tmp_inp, tmp_targets[0], tmp_targets[1]
                        else:
                            yield tmp_inp, tmp_targets
            
class SeqGen(keras.utils.Sequence):
    def __init__(self, bbox_util,batch_size,train_lines, val_lines, image_size,num_classes):
        self.gen = Generator(bbox_util,batch_size,train_lines, val_lines, image_size,num_classes)
        self.batch_size = batch_size
        self.train_lines = train_lines

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            x, y = self.gen.generate(True,eager=False)
            batch_x.append(x)
            batch_y.append(y)
            
        return np.array(batch_x), np.array(batch_y)
    
    def __len__(self):
        return self.train_lines / self.batch_size

