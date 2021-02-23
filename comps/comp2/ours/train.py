import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from efficientdet_training import Generator, SeqGen
from losses import focal,smooth_l1 
from model import EfficientDet, get_efficientdet_info
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from utils.utils import BBoxUtility, ModelCheckpoint
from utils.anchors import get_anchors
from functools import partial
from tqdm import tqdm
import time
import os

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

freeze_layers = [226, 328, 328, 373, 463, 463, 655, 802]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

if __name__ == "__main__":
    phi = 1
    annotation_path = '2007_train.txt'

    classes_path = 'model_data/voc_classes.txt' 
    class_names = get_classes(classes_path)
    NUM_CLASSES = len(class_names)  

    model = EfficientDet(phi,num_classes=NUM_CLASSES)
    # priors = get_anchors(image_sizes[phi])
    priors = get_anchors(get_efficientdet_info(phi)['image_size'])
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    logging = TensorBoard(log_dir="logs")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    checkpoint = ModelCheckpoint('logs/phi1-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # for i in range(freeze_layers[phi]):
    for i in range(get_efficientdet_info(phi)['freeze_layers']):
        model.layers[i].trainable = False

    if True:
        BATCH_SIZE = 16
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        # gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
        #                 (image_sizes[phi], image_sizes[phi]),NUM_CLASSES)
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        get_efficientdet_info(phi)['image_shape'],NUM_CLASSES)
                
        model.compile(loss={
                    'regressor'    : smooth_l1(),
                    'classifier'    : focal()
                },optimizer=keras.optimizers.Adam(Lr)
        )   
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        model.fit(
                gen.generate(True,eager=False), 
                steps_per_epoch=max(1, num_train//BATCH_SIZE),
                validation_data=gen.generate(False,eager=False), 
                validation_steps=max(1, num_val//BATCH_SIZE),
                epochs=Freeze_Epoch, 
                verbose=1,
                initial_epoch=Init_Epoch ,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )

    # for i in range(freeze_layers[phi]):
    for i in range(get_efficientdet_info(phi)['freeze_layers']):
        model.layers[i].trainable = True

    if True:
        BATCH_SIZE = 16
        Lr = 5e-5
        Freeze_Epoch = 50
        Epoch = 100
        # gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
        #                 (image_sizes[phi], image_sizes[phi]),NUM_CLASSES)
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        get_efficientdet_info(phi)['image_shape'],NUM_CLASSES)
                
        model.compile(loss={
                    'regressor'    : smooth_l1(),
                    'classifier': focal()
                },optimizer=keras.optimizers.Adam(Lr)
        )   
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        model.fit(
                gen.generate(True,eager=False), 
                steps_per_epoch=max(1, num_train//BATCH_SIZE),
                validation_data=gen.generate(False,eager=False), 
                validation_steps=max(1, num_val//BATCH_SIZE),
                epochs=Freeze_Epoch, 
                verbose=1,
                initial_epoch=Init_Epoch ,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )
