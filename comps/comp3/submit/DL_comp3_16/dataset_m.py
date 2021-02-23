import os
import time
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

from config import Config
from utils import adjust_data_range

def load_imgpaths(dir_path, split=False):

	files = os.listdir(dir_path)
	paths = [dir_path + file for file in files]
	random.shuffle(paths)
	if split:
		train_paths, eval_paths = \
		train_test_split(paths, test_size=0.05, random_state=Config.random_seed)
		return train_paths, eval_paths
	return paths

def load_and_preprocess(img_path):

    # img_str = tf.io.read_file(img_path)
    # img = tf.image.decode_and_crop_jpeg(img_str, Config.data_crop, channels=3)
	img_str = tf.io.read_file(img_path)
	img = tf.image.decode_jpeg(img_str, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, Config.data_size)
	img = tf.image.convert_image_dtype(img, tf.uint8)
	# img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
	# print(img)
	return img

def write_tfrecord(img_paths, tfrec_path):

	path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
	image_ds = path_ds.map(load_and_preprocess, num_parallel_calls = Config.parallel_threads) 
	proto_ds = image_ds.map(tf.io.serialize_tensor)
	tfrec = tf.data.experimental.TFRecordWriter(tfrec_path)
	tfrec_op = tfrec.write(proto_ds)

def prepare_tfrecords():

	print("Preparing tfrecords")
	paths = load_imgpaths(Config.data_dir_path)
	if len(paths)==2:
		write_tfrecord(paths[0], Config.tfrecord_dir+'train.tfrecord')
		write_tfrecord(paths[1], Config.tfrecord_dir+'test.tfrecord')
	else:
		write_tfrecord(paths, Config.tfrecord_dir+'train.tfrecord')

def parse_fn(tfrecord):

	result_img = tf.io.parse_tensor(tfrecord, out_type=tf.uint8)
	result_img = tf.reshape(result_img, Config.img_shape)
	result_img = tf.cast(result_img, tf.float32)
	result_img = adjust_data_range(result_img, drange_in=[0,255], drange_out=[-1, 1])
	return result_img

def parse_fn_img(tfrecord):

	result_img = tf.io.parse_tensor(tfrecord, out_type=tf.uint8)
	result_img = tf.reshape(result_img, Config.img_shape)
	result_img = tf.cast(result_img, tf.float32)
	result_img = adjust_data_range(result_img, drange_in=[0,255], drange_out=[-1, 1])
	return result_img

def parse_fn_cap(tfrecord):

	result_img = tf.io.parse_tensor(tfrecord, out_type=tf.float32)
	result_img = tf.reshape(result_img, Config.embed_shape)
	result_img = tf.cast(result_img, tf.float32)
	result_img = adjust_data_range(result_img, drange_in=[0,255], drange_out=[-1, 1])
	return result_img

def prepare_dataset(tfrecord_file):
	dataset = tf.data.TFRecordDataset(tfrecord_file)
	dataset = dataset.map(map_func=parse_fn, num_parallel_calls= Config.parallel_threads)
	dataset = dataset.shuffle(buffer_size=Config.total_training_imgs, reshuffle_each_iteration=True)
	return dataset

def process_dataset(dataset):
	dataset = dataset.batch(batch_size=Config.global_batchsize, drop_remainder=True)
	dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
	return dataset

def prepare_dataset_img(tfrecord_file):

	dataset = tf.data.TFRecordDataset(tfrecord_file)
	dataset = dataset.map(map_func=parse_fn_img, num_parallel_calls= Config.parallel_threads)
	dataset = dataset.shuffle(buffer_size=Config.total_training_imgs, reshuffle_each_iteration=True)
	# dataset = dataset.batch(batch_size=Config.global_batchsize, drop_remainder=True)
	# dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
	return dataset

def prepare_dataset_cap(tfrecord_file):

	dataset = tf.data.TFRecordDataset(tfrecord_file)
	dataset = dataset.map(map_func=parse_fn_cap, num_parallel_calls= Config.parallel_threads)
	dataset = dataset.shuffle(buffer_size=Config.total_training_imgs, reshuffle_each_iteration=True)
	# dataset = dataset.batch(batch_size=Config.global_batchsize, drop_remainder=True)
	# dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
	return dataset

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_t2i():
	def map_path(img):
		return Config.data_dir_path + img

	t2i_path = Config.text2ImgDataSplit_path
	t2i = load_obj(t2i_path)
	# Append taget image paths
	t2i['ImagePath'] = t2i['ImagePath'].apply(map_path)
	return t2i['Captions'].tolist(), t2i['ImagePath'].tolist()

def cap_preprocess_map(cap):
	return tf.constant(cap)

def cap_img_preprocess_map(cap, img_name):
	img = load_and_preprocess(img_name)
	return cap, img

def map2dataset_t2i(caps, imgs):
	# Captions and Images
	ds_cap_img = tf.data.Dataset.from_tensor_slices((caps, imgs))
	ds_cap_img = ds_cap_img.map(cap_img_preprocess_map, num_parallel_calls = Config.parallel_threads) 

	# Captions
	ds_cap = tf.data.Dataset.from_tensor_slices(caps)
	# ds_cap = ds_cap.map(cap_preprocess_map, num_parallel_calls = Config.parallel_threads) 

	# Images
	ds_img = tf.data.Dataset.from_tensor_slices(imgs)
	ds_img = ds_img.map(load_and_preprocess, num_parallel_calls = Config.parallel_threads) 

	return ds_cap_img, ds_cap, ds_img

# # The following functions can be used to convert a value to a type compatible with tf.Example.
# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#     """Returns a float_list from a float / double."""
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def _int64_feature(value):
#     """Returns an int64_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def serialize_example(feature0, feature1):
#     """
#     Creates a tf.Example message ready to be written to a file.
#     """
#     # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
#     feature = {
#         'feature0': _int64_feature(feature0),
#         'feature1': _int64_feature(feature1),
#     }

#     # Create a Features message using tf.train.Example.
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    
#     return example_proto.SerializeToString()

# def tf_serialize_example(f0,f1):
#     tf_string = tf.py_function(
#         serialize_example,
#         (f0,f1),  # pass these args to the above function.
#         tf.string)      # the return type is `tf.string`.
#     return tf.reshape(tf_string, ()) # The result is a scalar

def ser(cap, img):
	return tf.io.serialize_tensor(cap), tf.io.serialize_tensor(img)

# @tf.function
def to_tfrecord(ds, tfrec_path):
	proto_ds = ds.map(tf.io.serialize_tensor)
	tfrec = tf.data.experimental.TFRecordWriter(tfrec_path)
	tfrec_op = tfrec.write(proto_ds)

def cap_img_to_tfrecord(ds, tfrec_path):
	proto_ds = ds.map(ser)
	tfrec = tf.data.experimental.TFRecordWriter(tfrec_path)
	tfrec_op = tfrec.write(proto_ds)

# @tf.function
def make_tfrecords():
	cap, img_path = load_t2i()
	ds_cap_img, ds_cap, ds_img = map2dataset_t2i(cap, img_path)

	print("Preparing Image tfrecord")
	to_tfrecord(ds_img, Config.tfrecord_dir+'img_train.tfrecord')

	# Bugs here
	# Refer to 
	# https://nthu-datalab.github.io/ml/labs/12-1_CNN/12-1_CNN.html
	# https://www.tensorflow.org/tutorials/load_data/tfrecord

	# print("Preparing Caption&Image tfrecord")
	# cap_img_to_tfrecord(ds_cap_img, Config.tfrecord_dir+'cap_img_train.tfrecord')
	print("Preparing Caption tfrecord")
	to_tfrecord(ds_cap, Config.tfrecord_dir+'cap_train.tfrecord')
	

def main():
	# tf.enable_eager_execution()
	start_time = time.time()
	prepare_tfrecords()
	make_tfrecords()
	end_time = time.time()
	print(end_time-start_time)

if __name__ == '__main__':
	main()
	# load_t2i()
