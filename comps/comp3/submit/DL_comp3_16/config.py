class Config:

	# dataset preparation config for CelebA dataset
	data_dir_path = '/home/weidagogo/chi-shen/comp3/DCGAN-TF2.0/datalab-cup3-reverse-image-caption-2020/102flowers/102flowers/'
	text2ImgDataSplit_path = '/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/train_split.pkl'
	tfrecord_dir = 'tfrecords/'  # dir where tfrecords are saved
	data_crop = [57, 21, 128, 128] # [crop-y(top-left y), crop-x(top-left x), crop-height, crop-width] (128 x 128 resolution images are taken)
	data_size = [128, 128]
	img_shape = (128, 128, 3) # image shape for Variational Autoencoder
	embed_shape = (4800,) # image shape for Variational Autoencoder
	parallel_threads = 4   # number of parallel calls for writing and reading tfrecord (Depends on no.of cpu cores)

	# batch_norm layer parameters
	momentum = 0.9
	epsilon  = 1e-5

	# DCGAN architecture 
	latent_dim = 4800
	disc_filters = [32, 64, 128, 256, 512]
	gen_filters  = [1024, 512, 256, 128, 64]

	#training parameters
	disc_lr = 0.0002  #learning rate of discriminator
	gen_lr = 0.0002   #learning rate of generator
	global_batchsize = 32   # Configure it based on available GPU memory
	# total_training_imgs = 7370
	total_training_imgs = 36850
	num_gpu = 1
	num_epochs = 1000000
	image_snapshot_freq = 1000  # Number of batches shown in between image_grid snapshots

	#Adam optimizer parameters
	beta1 = 0.5
	beta2 = 0.999

	random_seed = 42
	#results
	modelDir = 'model/'
	summaryDir = 'summaries/run4/'
	num_gen_imgs = 32   # number of images to generate
	grid_size = (4, 8)  # results are saved to an image grid of this size 
	results_dir = 'results/'

	#Generator is run twice for this run
