from efficientdet import EfficientDet
from PIL import Image
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

efficientdet = EfficientDet()

# while True:
# img = "./VOCdevkit_test/VOC2007/JPEGImages/000008.jpg"
# try:
#     image = Image.open(img)
# except:
#     print('Open Error! Try again!')
# else:
#     r_image = efficientdet.detect_image(image)
#     r_image.save("img/000008.jpg")

test_img_files = open('./pascal_voc_testing_data.txt')
test_img_dir = './VOCdevkit_test/VOC2007/JPEGImages/'
test_images = []

for line in test_img_files:
    line = line.strip()
    ss = line.split(' ')
    test_images.append(ss[0])

output_file = open('./test_prediction.txt', 'w')
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
        res = ""
        for i, c in enumerate(label):
            res += " %d %d %d %d %d %f" %(x0[i], x1[i], y0[i], y1[i], label[i], conf[i])
            
        #print("res: ", res)
        output_file.write(img_name + res + "\n")

output_file.close()