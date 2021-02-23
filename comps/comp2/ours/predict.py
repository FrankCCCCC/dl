from efficientdet import EfficientDet
from PIL import Image
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus[2], 'GPU')

efficientdet = EfficientDet()

test_img_files = open('./pascal_voc_testing_data.txt')
test_img_dir = './VOCdevkit_test/VOC2007/JPEGImages/'
test_images = []

for line in test_img_files:
    line = line.strip()
    ss = line.split(' ')
    test_images.append(ss[0])

output_file = open('./test_prediction.txt', 'w')

res_list = []

for img_name in test_images:
    img = test_img_dir + img_name
    #print(img)
    try:
        image = Image.open(img)
        #print(image)
    except:
        print('Can not find image!')
        continue
    else:
        x0, x1, y0, y1, label, conf = efficientdet.output_txt(image)
        #print("img: ", img, x0, x1, y0, y1, label, conf)
        res = ""
        for i, c in enumerate(label):
            res += " %d %d %d %d %d %f" %(x0[i], x1[i], y0[i], y1[i], label[i], conf[i])
            
        #print("res: ", res)
        res_list.append(img_name+res)


for i, c in enumerate(res_list):
    output_file.write(res_list[i] + "\n")

output_file.close()

# class BoxNet(models.Model):
#     def __init__(self, width, depth, num_anchors=9, separable_conv=True, freeze_bn=False, detect_quadrangle=False, **kwargs):
#     super(BoxNet, self).__init__(**kwargs)
#         self.width = width
#         self.depth = depth
#         self.num_anchors = num_anchors
#         self.separable_conv = separable_conv
#         self.detect_quadrangle = detect_quadrangle
#         num_values = 9 if detect_quadrangle else 4

#         if separable_conv:
#             kernel_initializer = {
#                 'depthwise_initializer': initializers.VarianceScaling(),
#                 'pointwise_initializer': initializers.VarianceScaling(),
#             }
#             self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'bias_initializer': 'zeros', **kernel_initializer) for i in range(depth)]
#             self.head = layers.SeparableConv2D(filters=num_anchors * num_values,name=f'{self.name}/box-predict', 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'bias_initializer': 'zeros', **kernel_initializer)
        
#         else:
#             kernel_initializer = {
#                 'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
#             }
#             self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'bias_initializer': 'zeros', **kernel_initializer) for i in range(depth)]
#             self.head = layers.SeparableConv2D(filters=num_anchors * num_values,name=f'{self.name}/box-predict', 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'bias_initializer': 'zeros', **kernel_initializer)
        
#         self.bns = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/box-{i}-bn-{j}') for j inrange(3, 8)]for i in range(depth)]
#         self.level = 0
    
#     def call(self, inputs, **kwargs):
#         feature, level = inputs
#         for i in range(self.depth):
#                 feature = self.convs[i](feature)
#                 feature = self.bns[i][self.level](feature)
#                 feature = layers.Lambda(lambda x: tf.nn.swish(x))(feature)
#         outputs = self.head(feature)
#         outputs = layers.Reshape((-1, num_values))(outputs)
#         outputs = layers.Activation('sigmoid')(outputs)
#         self.level += 1
#         return outputs
# 