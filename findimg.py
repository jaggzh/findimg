#!/usr/bin/python2.7
# findimg.py
# jaggz.h is at gmail.com and jaggz on Freenode IRC in ##machinelearning

# This software attempts to find images in your global set of images
#  based on some labeled folders in imgdat/labeled/
# For example, you might have a folder of images imgdat/labeled/dogs
# It attempts to distinguish between your labeled sets and your
#  global set of images.
# See config.py for setup help. It doesn't require much

# Run with findimg.py -h for help

import numpy
#from keras.constraints import maxnorm
#from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import maxnorm
from keras.models import Sequential, Model # , load_weights, save_weights
from keras.layers import Dense, merge, Reshape, UpSampling2D, Flatten, Convolution2D, MaxPooling2D, Input, ZeroPadding2D, Activation, Dropout, Deconvolution2D
#from keras.layers import Deconvolution2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.layer_utils import print_summary
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils
import keras
import numpy as np
#from keras import backend as K
#K.set_image_dim_ordering('th')
from bh.util import exit, pf, pfp, pfl, pfpl, eprint, vprint
from bh.bansi import *
import config
from hashlib import sha256
import random, re, os, time, sys, math, shutil, itertools, argparse
from os import listdir
from os.path import isdir, isfile, join
from PIL import Image
import PIL

seed = 7; numpy.random.seed(seed)
indim = 64
args = None
allglobs = None
globsets = None
lbld = None
test_frac = .10
val_frac = .10
model = None
labeled_frac = .5 # Fraction of generated images coming from labeled set(s)
label_names = []
label_ints = dict()
label_onehots = dict()
grayscale=False
channels = 1 if grayscale else 3
lbl_cnt = None
load_weights = True
weight_store = "dat/weights.h5"
save_weights = True
train_epochs = 100
train_samples = 250
save_weight_secs = 20
start_time = time.time()
last_epoch_time = None
fit_start_time = None
checkpoint_epoch = None

def load_globs():
	global globsets
	global allglobs
	f = open(config.file_pics_glob_files)
	globs = [x.rstrip('\r\n') for x in f.readlines()]
	f.close()
	filtered = []
	prog = re.compile('\.(jpg|gif|tif|tiff|png|bmp)$', re.IGNORECASE)
	for fn in globs:
		if prog.search(fn):
			filtered.append(fn)
	#for x in globs: print(x)
	globs = filtered
	random.shuffle(globs)
	allglobs = globs
	globsets = split_set(globs)
def load_imagenames_recursively(path):
	files = []
	for path, dirs, names in os.walk(os.path.join(path, "images")):
		#print "Path: ", path
		#print "Names: ", names
		for fn in names:
			if not fn.startswith("."):
				files.append(os.path.join(path, fn))
	#print "File list: ", files
	return files
def load_labeled():
	global lbld
	global label_names
	global label_onehots
	global label_ints
	global lbl_cnt
	lbld = dict()
	for path, dirs, names in os.walk(config.dir_labeled_base):
		for labeldir in dirs:
			label_names.append(labeldir)
			files = load_imagenames_recursively(os.path.join(path, labeldir))
			#print "Files returned: ", files
			#exit(0)
			random.shuffle(files)
			sets = split_set(files)
			for ttv in sets.keys(): # train, test, val
				if not lbld.has_key(ttv):
					lbld[ttv]=[]
				for img in sets[ttv]: # filenames
					#print("Appending img,labeldir:", img, labeldir)
					lbld[ttv].append([img, labeldir])
			dirs.remove(labeldir)
	#print "Label names:", label_names

	lbl_cnt = len(label_names)+1 # add one for global label as 0
	pf("Total labels, including global:", lbl_cnt)
	i=1 # Start at 1 to skip global image label at 0
	label_names.sort()
	pf("Number of non-global labels: ", len(label_names))
	for lbl in label_names:
		label_onehots[lbl] = int_to_onehot(i)
		label_ints[lbl] = i
		i += 1
	#pf("Labeled sets:", lbld);
	#pf("Labeled set:", lbld['train']);
	#print "Label onehots:", label_onehots
	#exit(0)
	#for x in globs: print(x)
def int_to_onehot(i):
	#arr = np.array([0,2,3])
	#pf("arr shape:", arr.shape)
	#arr = np_utils.to_categorical(arr)
	#pf("categorical shape:", arr.shape)
	#pf("Array: ", arr)
	arr = np.zeros((lbl_cnt), dtype='float32')
	arr[i]=1
	arr=arr.reshape((1,lbl_cnt))
	#pf("Final Array: ", arr)
	#pf("Final Array shape: ", arr.shape)
	return arr
def split_set(arr):
	lenarr = len(arr)
	aset = dict()
	#print("Array length:", lenarr)
	#print("  test len:", int(lenarr*test_frac))
	#print("   val len:", int(lenarr*(test_frac+val_frac)))
	#print(" train len:", int(lenarr*(test_frac+val_frac)))
	aset['test'] = arr[ : int(lenarr*test_frac) ]
	aset['val'] = arr[ int(lenarr*test_frac) : int(lenarr*(test_frac+val_frac)) ]
	aset['train'] = arr[ int(lenarr*(test_frac+val_frac)) : ]
	return aset
def load_args():
	global args
	parser = argparse.ArgumentParser(description='Find matching images')
	#parser.add_argument('-f', '--file', default=None, help='Use file as input test image')
	parser.add_argument('-H', '--threshold', default=.9, help='Threshold (0-1.0) above which matches will display with -f')
	parser.add_argument('-t', '--train', action='store_true', help='Train')
	parser.add_argument('-f', '--find', default=None, help='Find images matching label')
	parser.add_argument('-v', '--verbose', default=0, help='Set verbosity 0, 1, etc.')
	parser.add_argument('-c', '--showcaching', action='store_true', help='Show image caching process')
	args=parser.parse_args()
class SaveWeights(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		global last_epoch_time
		#plt.pause(0.01) # Calls matplotlib's event loop
		if time.time()-last_epoch_time > save_weight_secs:
			last_epoch_time = time.time()
			pfp("Saving weights, timed (", save_weight_secs, "s).  Time elapsed: ",
					int(time.time()-start_time), "s.  Fit time elapsed: ",
					int(time.time()-fit_start_time), "s.",
				)
			save_weights()
		return
def init():
	load_args()
	load_globs() # creates globsets[test/train/val]
	load_labeled() # creates lbld['labelname'][test/train/val]
	global checkpoint_epoch
	checkpoint_epoch = SaveWeights()
def conv2d_leaky(layer, filters=None, filtdim=2, ss=(1,1), leakyalpha=.2, border='same', name=None):
	newlayer = Convolution2D(
			filters,
			filtdim,
			filtdim,
			border_mode=border,
			subsample=ss,
			name=name
		)(layer)
	newlayer = LeakyReLU(alpha=leakyalpha)(newlayer)
	return newlayer

def show_a_prediction(label="", model=None, imgsetname=None):
	return
	#if imgsetname == None: raise ValueError("Requires model and imgsetname")
	#[ximg,yimg] = next(source_imgs(imgsetname))
	#prediction = model.predict(ximg, batch_size=1, verbose=1)
	#view_img("("+label+" In)", ximg[0], show=True)
	#view_img("("+label+" GndTruth)", yimg[0], show=True)
	#view_img("("+label+" Pred)", prediction[0])
	#raw_input('ENTER');

def label_to_onehot(lbl):
	return label_onehots[lbl]
def prep_img(fn):
	img = get_cached(fn)
	#if img:
		#if args.showcaching:
			#pfp(bgre, "Using cached image: ", fn, rst)
	if img == None:
		#pf("Preparing image:", fn)
		try:
			img = load_img(fn) #, grayscale=grayscale)
		except:
			pf("ERROR: Error loading image:", fn, "\n", sys.exc_info()[0])
			return None
		if img == None:
			pfp(bred, "Error loading image: ", fn, rst)
			return None
		else:
			if args.showcaching:
				pfp(bcya, "Caching image: ", fn, rst)
			#img = img.resize((indim,indim), resample=PIL.Image.LANCZOS)
			img.thumbnail((indim,indim), PIL.Image.ANTIALIAS)
			bg = Image.new('RGB', (indim,indim), (0, 0, 0))
			bg.paste(
			    	img,
					(int((indim - img.size[0]) / 2), int((indim - img.size[1]) / 2))
				)
			img=bg
			store_cache(fn, img)

	img = img_to_array(img)  # Numpy array with shape (?, width, height)
	img = img/255.0
	img = img.reshape((1,)+img.shape)
	#pf("Prepared shape:", img.shape)
	return img
def get_cached(fn):
	cdir, d2, hfull = fn2hashpath(fn)
	hfull = hfull + ".jpg"
	if not os.path.exists(hfull): return None
	#pf("Loading cached image:", hfull)
	try:
		f = Image.open(hfull)
	except IOError as e:
		pf("I/O error({0}): {1}".format(e.errno, e.strerror))
		return None
	except: #handle other exceptions such as attribute errors
		pf("Unexpected error:", sys.exc_info()[0])
		return None
	return f

def store_cache(fn, img):
	cdir, d2, hfull = fn2hashpath(fn)
	hfull = hfull + ".jpg"
	if not isdir(d2):
		os.makedirs(d2)
	try:
		#pf(bred, "Storing img to cache:", hfull, rst)
		img.save(hfull, "JPEG", quality=87)
	except IOError as e:
		pf("I/O error({0}): {1}".format(e.errno, e.strerror))
		return None
	except: #handle other exceptions such as attribute errors
		pf("Unexpected error:", sys.exc_info()[0])
		return None
#output:
# caching cache/83/83a1ff93f5f6ea500bfd5aec008507...
# Unexpected error: <type 'exceptions.ValueError'>


def fn2hashpath(fn):
	m=sha256(fn).hexdigest()
	#pf(bgblu, whi, "HASHED", fn, "->", m, rst)
	m2=m[ :2]
	cac=config.dir_cache
	d2=os.path.join(cac, m2)
	hpath=os.path.join(d2, m)
	return cac, d2, hpath

def source_imgs(imgsetname): # imgsetname='train','val','test'
	while True:
		#pf("source_imgs(", imgsetname, ") called")
		if (random.random() < labeled_frac):
			# choose from labeled set
			fn_n_label_i = random.randint(0, len(lbld[imgsetname])-1)
			fn_n_label = lbld[imgsetname][fn_n_label_i]
			ximg = prep_img(fn_n_label[0])
			if ximg == None:
				lbld.pop(fn_n_label_i)
				continue
			lbl = label_to_onehot(fn_n_label[1])
			#pf("LBLD:", lbl, fn_n_label[0])
		else:
			#print globsets;
			fni = random.randint(0, len(globsets[imgsetname])-1)
			fn = globsets[imgsetname][fni]
			#pf("GLOB fn", fn)
			ximg = prep_img(fn)
			if ximg == None:
				globsets[imgsetname].pop(fni)
				continue
			lbl = int_to_onehot(0)   # global is 0!
			#pf("GLOB:", lbl, fn)
		#pf("Yielding ximg shape = ", ximg.shape, sep='')
		#pf("Y shape", lbl.shape)
		#pf("Yielding: y=", lbl, " data ", ximg, sep='')
		yield ximg, lbl

def make_model():
	x = inputs = Input(shape = (channels, indim, indim))
#	# 32
#	x = conv2d_leaky(x, filters=32, filtdim=2, ss=(1,1), border='valid', name='first')
#	x = conv2d_leaky(x, filters=32, filtdim=2, ss=(2,2), border='valid', name='first')
#	x = Dropout(.2)(x)
#	# 16
#	x = conv2d_leaky(x, filters=64, filtdim=2, ss=(1,1), border='valid', name='first')
#	x = conv2d_leaky(x, filters=64, filtdim=2, ss=(2,2), border='valid', name='first')
#	# 8
#	x = conv2d_leaky(x, filters=64, filtdim=2, ss=(1,1), border='valid', name='first')
#	x = conv2d_leaky(x, filters=64, filtdim=2, ss=(2,2), border='valid', name='first')
	# 64
	filt=64
	x = Convolution2D(filt, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = Dropout(0.2)(x)
	x = Convolution2D(filt, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = MaxPooling2D((2, 2))(x)
	# 32 or something
	x = Convolution2D(filt*2, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = Convolution2D(filt*2, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = MaxPooling2D((2, 2))(x)
	# 16 or something
	x = Convolution2D(filt*4, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = Convolution2D(filt*4, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = MaxPooling2D((2, 2))(x)
	# 8 or something
	x = Convolution2D(filt*8, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = Convolution2D(filt*8, 2, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(3))(x)
	x = MaxPooling2D((2, 2))(x)
	# 4 or something
	x = Flatten()(x)
	x = Dense(512, activation='relu', W_constraint=maxnorm(3))(x)
	x = Dropout(0.1)(x)
	num_classes=lbl_cnt
	x = Dense(num_classes, activation='softmax')(x)
	model = Model(input=inputs, output=x)
	# Compile model
	epochs = train_epochs
	lrate = 0.001
	decay = lrate/epochs
	#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	sgd = SGD(lr=lrate, momentum=0.9, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	print(model.summary())
	if load_weights and isfile(weight_store):
		pf("Loading weights")
		model.load_weights(weight_store)
	return model

def train_model(loops=1, epochs=train_epochs, samples=train_samples, view=5, save=True):
	gen_epochs = epochs
	gen_samples = samples
	gen_fits = 0
	######### Train generator with normal convolutions
	for gen_fits in range(loops):
		pf(bcya, "Calling generator.fit_generator()", rst)
		pf(bcya, "Loops:", loops, "Epochs:", gen_epochs, "Samples:", gen_samples, rst)
		pf(bcya, "Fit runs:", gen_fits, " Total epochs now:", gen_fits*gen_epochs, "Total samples now:", gen_fits*gen_epochs*gen_samples, rst)
		#'''
		if gen_epochs>0:
			global fit_start_time, last_epoch_time
			fit_start_time = time.time()
			last_epoch_time = time.time()
			model.fit_generator(source_imgs('train'),
					samples_per_epoch=gen_samples,
					nb_epoch=gen_epochs,
					verbose=2,
					nb_worker=1,
					validation_data=source_imgs('test'),
					nb_val_samples=20,
					callbacks=[checkpoint_epoch],
				)
			gen_fits += 1
			if save: save_weights()
		if view:
			for i in range(3):
				show_a_prediction(label="Gen", model=model, imgsetname='test')

def save_weights():
	model.save_weights(weight_store)
	pf("Saved generator weights.")

def find_imgs(lbl):
	lbli = label_ints[lbl]
	count = 0
	globcount = len(allglobs)
	matches = []
	fni = -1
	while fni+1 < len(allglobs):
		fni = fni+1
		fn = allglobs[fni]
	#ml = lbld['train']
	#for fn in ml:
		##fn = fn[0]
		#pf("file:", fn)
		count = count+1
		img = prep_img(fn)
		if img == None:
			allglobs.pop(fni)
			fni = fni-1
			pf("ERROR: Removed global fn:", fn)
			continue
		y = model.predict(img, batch_size=1, verbose=0)
		matches.append([fn, y[0][lbli]])
		if y[0][lbli] >= args.threshold:
			pf("Predict:", fn, " -> y:", y)
			pf("  y[0][", lbli, "] = ", y[0][lbli])
			pf("  fn:", fn)
		if (count%100) == 0:
			pf("Searched", count, "of", globcount)
	matches = sorted(matches, key=sort_y, reverse=True)
	pf("Results:")
	for m in matches:
		pf(m[1], m[0])
def sort_y(a):
	return a[1]

init()
model=make_model()
if args.train:
	train_model()
elif args.find != None:
	find_imgs(args.find)

# vim:ts=4 ai
