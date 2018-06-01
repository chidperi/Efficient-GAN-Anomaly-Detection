import numpy as np
import pandas as pd
import os
# import pathlib
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.image as mpimg

def downsample_data():
	work_folder_path = os.getcwd()
	csv_folder_path = work_folder_path+'/csv_files'

	num_of_files = len(os.listdir(csv_folder_path))
	flight_data_list = [0]*num_of_files
	flight_label_list = [0]*num_of_files
	window_size = 2000
	for root,dirs,files in os.walk(csv_folder_path):
		for file in files:
			file_noext, _ = os.path.splitext(file)
			file_num = int(file_noext[14:])
			df = pd.read_csv(root+'/'+file)
			df_array = df.as_matrix()
			df_data = df_array[:,:5]
			df_label = df_array[:,-1]
			num_of_samples = df_data.shape[0]/window_size
			data_list = [0]*num_of_samples
			label_list = [0]*num_of_samples
			for idx in xrange(num_of_samples):
				data_list[idx] = df_data[idx*window_size:(idx+1)*window_size:400]
				label_list[idx] = max(df_label[idx*window_size:(idx+1)*window_size])

			data_list_stack = np.vstack(data_list)
			flight_data = data_list_stack.reshape(num_of_samples, data_list[0].shape[0], data_list[0].shape[1])
			flight_data_list[file_num-1] = flight_data

			label_list_stack = np.vstack(label_list)
			# flight_label = label_list_stack.reshape(num_of_samples,)
			flight_label = label_list_stack
			flight_label_list[file_num-1] = flight_label

		flight_data_list_stack = np.vstack(flight_data_list)
		flight_label_list_stack = np.vstack(flight_label_list)
		flight_label_list_stack = flight_label_list_stack.reshape(flight_label_list_stack.shape[0],)

	normal_idx =  np.where(flight_label_list_stack==0)[0]
	abnormal_idx =	np.where(flight_label_list_stack==1)[0]
	train_idx = np.concatenate((normal_idx[:1500], abnormal_idx[:1500]), axis=0)
	test_idx  =	 np.concatenate((normal_idx[1500:2000], abnormal_idx[1500:2000]), axis=0)

	train_data = flight_data_list_stack[train_idx,:,:]
	train_label = flight_label_list_stack[train_idx]

	test_data = flight_data_list_stack[test_idx,:,:]
	test_label = flight_label_list_stack[test_idx]

	return train_data, train_label, test_data, test_label

def original_data():
	work_folder_path = os.getcwd()
	csv_folder_path = work_folder_path+'/csv_files'

	num_of_files = len(os.listdir(csv_folder_path))
	flight_data_list = [0]*num_of_files
	flight_label_list = [0]*num_of_files
	window_size = 2000
	for root,dirs,files in os.walk(csv_folder_path):
		for file in files:
			file_noext, _ = os.path.splitext(file)
			file_num = int(file_noext[14:])
			df = pd.read_csv(root+'/'+file)
			df_array = df.as_matrix()
			df_data = df_array[:,:5]
			df_label = df_array[:,-1]
			num_of_samples = df_data.shape[0]/window_size
			data_list = [0]*num_of_samples
			label_list = [0]*num_of_samples
			for idx in xrange(num_of_samples):
				data_list[idx] = df_data[idx*window_size:(idx+1)*window_size]
				label_list[idx] = max(df_label[idx*window_size:(idx+1)*window_size])

			data_list_stack = np.vstack(data_list)
			flight_data = data_list_stack.reshape(num_of_samples, data_list[0].shape[0], data_list[0].shape[1])
			flight_data_list[file_num-1] = flight_data

			label_list_stack = np.vstack(label_list)
			# flight_label = label_list_stack.reshape(num_of_samples,)
			flight_label = label_list_stack
			flight_label_list[file_num-1] = flight_label

		flight_data_list_stack = np.vstack(flight_data_list)
		flight_label_list_stack = np.vstack(flight_label_list)
		flight_label_list_stack = flight_label_list_stack.reshape(flight_label_list_stack.shape[0],)

	normal_idx =  np.where(flight_label_list_stack==0)[0]
	abnormal_idx =	np.where(flight_label_list_stack==1)[0]
	train_idx = np.concatenate((normal_idx[:1500], abnormal_idx[:1500]), axis=0)
	test_idx  =	 np.concatenate((normal_idx[1500:2000], abnormal_idx[1500:2000]), axis=0)

	train_data = flight_data_list_stack[train_idx,:,:]
	train_label = flight_label_list_stack[train_idx]

	test_data = flight_data_list_stack[test_idx,:,:]
	test_label = flight_label_list_stack[test_idx]

	return train_data, train_label, test_data, test_label

def normalize_flight_data(data):
	data = data.astype('float32')
	part_data = data[:,:2]
	part_data_mean = np.mean(part_data, axis=0)
	part_data_std = np.std(part_data, axis=0)
	part_data = (part_data-part_data_mean)/part_data_std
	data[:,:2] = part_data

	return data

def get_aircraft_data(aircraft_folder_path):
	# num_of_files = len(os.listdir(aircraft_folder_path))
	# flight_data_list = [0]*num_of_files
	flight_data_list = []
	# flight_label_list = [0]*num_of_files
	flight_label_list = []
	window_size = 2000
	for root,dirs,files in os.walk(aircraft_folder_path):
		for file in files:
			# file_noext, _ = os.path.splitext(file)
			# file_num = int(re.findall('\((.*?)\)', file_noext)[0])
			df = pd.read_csv(root+'/'+file)
			df_array = df.as_matrix()
			df_data = df_array[:,:5]
			df_label = df_array[:,-1]
			num_of_samples = int(df_data.shape[0]/window_size)
			# data_list = [0]*num_of_samples
			data_list = []
			# label_list = [0]*num_of_samples
			label_list = []
			for idx in range(num_of_samples):
				# data_list[idx] = df_data[idx*window_size:(idx+1)*window_size]
				data_list.append(df_data[idx*window_size:(idx+1)*window_size])
				# label_list[idx] = max(df_label[idx*window_size:(idx+1)*window_size])
				# label_list[idx] = sum(df_label[idx*window_size:(idx+1)*window_size])
				label_list.append(max(df_label[idx*window_size:(idx+1)*window_size]))

			data_list_stack = np.vstack(data_list)
			data_list_stack = normalize_flight_data(data_list_stack)
			flight_data = data_list_stack.reshape(num_of_samples, data_list[0].shape[0], data_list[0].shape[1])
			# flight_data_list[file_num-1] = flight_data
			flight_data_list.append(flight_data)

			label_list_stack = np.vstack(label_list)
			# flight_label = label_list_stack.reshape(num_of_samples,)
			flight_label = label_list_stack
			# flight_label_list[file_num-1] = flight_label
			flight_label_list.append(flight_label)

		flight_data_list_stack = np.vstack(flight_data_list)
		flight_label_list_stack = np.vstack(flight_label_list)
		# flight_label_list_stack = flight_label_list_stack.reshape(flight_label_list_stack.shape[0],)

	return flight_data_list_stack, flight_label_list_stack

def get_aircraft_feature(aircraft_folder_path):
	# num_of_files = len(os.listdir(aircraft_folder_path))
	# flight_data_list = [0]*num_of_files
	flight_feat_list = []
	# flight_label_list = [0]*num_of_files
	flight_label_list = []
	# window_size = 2000
	for root,dirs,files in os.walk(aircraft_folder_path):
		for file in files:
			# file_noext, _ = os.path.splitext(file)
			# file_num = int(re.findall('\((.*?)\)', file_noext)[0])
			df = pd.read_csv(root+'/'+file)
			df_array = df.as_matrix()
			df_feat = df_array[:,:70]
			df_label = df_array[:,-1]

			flight_feat_list.append(df_feat)
			flight_label_list.append(df_label)

		flight_data_list_stack = np.vstack(flight_feat_list)
		flight_label_list_stack = np.hstack(flight_label_list)

	return flight_data_list_stack, flight_label_list_stack

def cross_validation(train_aircraft_num, test_aircraft_num):

	train_test_data_folder = "train_test_data/aircraft_"+str(test_aircraft_num[0])
	if not os.path.isdir(train_test_data_folder):
		# pathlib.path(train_test_data_folder).mkdir(parents=True)
		os.makedirs(train_test_data_folder)

		work_folder_path = os.getcwd()
		csv_folder_path = work_folder_path+'/CSVs'

		train_flight_data = []
		train_flight_label = []
		for aircraft_num in train_aircraft_num:
			aircraft_folder_path = csv_folder_path+'/Aircraft '+str(aircraft_num)
			data, label = get_aircraft_data(aircraft_folder_path) 
			train_flight_data.append(data)
			train_flight_label.append(label)

		test_flight_data = []
		test_flight_label = []
		for aircraft_num in test_aircraft_num:
			aircraft_folder_path = csv_folder_path+'/Aircraft '+str(aircraft_num)
			data, label = get_aircraft_data(aircraft_folder_path)
			test_flight_data.append(data)
			test_flight_label.append(label)
		
		train_data = np.vstack(train_flight_data)
		train_label = np.vstack(train_flight_label)
		train_label = train_label.reshape(train_label.shape[0],)

		test_data = np.vstack(test_flight_data)
		test_label = np.vstack(test_flight_label)
		test_label = test_label.reshape(test_label.shape[0],)

		# normal_idx =	np.where(test_label==0)[0]
		# abnormal_idx =	np.where(test_label==1)[0]
		# print normal_idx.shape[0], abnormal_idx.shape[0]

		# def to_percent(y, position):
		#	  # Ignore the passed in position. This has the effect of scaling the default
		#	  # tick locations.
		#	  s = str(100 * y)

		#	  # The percent symbol needs escaping in latex
		#	  if matplotlib.rcParams['text.usetex'] is True:
		#		  return s + r'$\%$'
		#	  else:
		#		  return s + '%'

		# plt.hist(test_label, bins=100, weights= [1./ test_label.shape[0]] * test_label.shape[0])
		# plt.title("Aircraft "+str(test_aircraft_num[0]))

		# # Create the formatter using the function to_percent. This multiplies all the
		# # default labels by 100, making them all percentages
		# formatter = FuncFormatter(to_percent)

		# # Set the formatter
		# plt.ylim((0,1))
		# plt.gca().yaxis.set_major_formatter(formatter)

		# plt.show()

		np.save(train_test_data_folder+"/train_data.npy", train_data)
		np.save(train_test_data_folder+"/train_label.npy", train_label)
		np.save(train_test_data_folder+"/test_data.npy", test_data)
		np.save(train_test_data_folder+"/test_label.npy", test_label)
	else:
		train_data = np.load(train_test_data_folder+"/train_data.npy")
		train_label = np.load(train_test_data_folder+"/train_label.npy")
		test_data = np.load(train_test_data_folder+"/test_data.npy")
		test_label = np.load(train_test_data_folder+"/test_label.npy")

	return train_data, train_label, test_data, test_label

def convert_tsdata_to_image(data, image_folder):

	img_all = []
	for img_idx in range(0,data.shape[0]):
		img = []
		for chnl_idx in range(0, data.shape[2]):
			plt_data = list(data[img_idx, :, chnl_idx].reshape(data.shape[1],1))
			# plt_data = data[img_idx, :, chnl_idx].reshape(data.shape[1],1)
			plt.plot(plt_data)
			# plt.ylim((-40,20))
			# plt.ylim((-20,20))
			plt.ylim((-10,10))
			plt.axis('off')
			# plt.savefig(image_folder+'/{:05d}.png'.format(img_idx+1))
			plt.savefig('test.png')
			plt.close()
			# plt.show()
			# img=mpimg.imread(image_folder+'/{:05d}.png'.format(img_idx+1))
			img1 = mpimg.imread('test.png')
			img2 = img1[3:477, 89:563, 0].astype(int)
			img2 = img2.reshape(img2.shape[0], img2.shape[1], 1)
	  
			if chnl_idx == 0:
				img = img2
			else:
				img = np.concatenate((img,img2), axis=2)
		img_all.append(img)
	img_all = np.array(img_all)

	np.save(image_folder+'/image_array.npy', img_all)

	return img_all

def cross_validation_on_image(train_aircraft_num, test_aircraft_num):
	train_test_data_folder = "train_test_data/aircraft_"+str(test_aircraft_num[0])
	if not os.path.isdir(train_test_data_folder):
		# pathlib.path(train_test_data_folder).mkdir(parents=True)
		os.makedirs(train_test_data_folder)
		train_data, train_label, test_data, test_label = cross_validation(train_aircraft_num, test_aircraft_num)
		np.save(train_test_data_folder+"/train_data.npy", train_data)
		np.save(train_test_data_folder+"/train_label.npy", train_label)
		np.save(train_test_data_folder+"/test_data.npy", test_data)
		np.save(train_test_data_folder+"/test_label.npy", test_label)
	else:
		train_data = np.load(train_test_data_folder+"/train_data.npy")
		train_label = np.load(train_test_data_folder+"/train_label.npy")
		test_data = np.load(train_test_data_folder+"/test_data.npy")
		test_label = np.load(train_test_data_folder+"/test_label.npy")

	train_image_folder = "train_test_image/aircraft_"+str(test_aircraft_num[0])+"/train"
	if not os.path.isdir(train_image_folder):
		os.makedirs(train_image_folder)

	test_image_folder = "train_test_image/aircraft_"+str(test_aircraft_num[0])+"/test"
	if not os.path.isdir(test_image_folder):
		os.makedirs(test_image_folder)

	train_image = convert_tsdata_to_image(train_data, train_image_folder)
	test_image = convert_tsdata_to_image(test_data, test_image_folder)

def cross_validation_on_DA_features(train_aircraft_num, test_aircraft_num):
	train_test_data_folder = "/home/chidperi/Projects/Efficient-GAN-Anomaly-Detection/data/train_test_data/aircraft_"+str(test_aircraft_num[0])
	if not os.path.isdir(train_test_data_folder):
		# pathlib.path(train_test_data_folder).mkdir(parents=True)
		os.makedirs(train_test_data_folder)

		work_folder_path = os.getcwd()
		csv_folder_path = work_folder_path+'/CSV_FEATURES'

		train_flight_data = []
		train_flight_label = []
		for aircraft_num in train_aircraft_num:
			aircraft_folder_path = csv_folder_path+'/Aircraft '+str(aircraft_num)
			data, label = get_aircraft_feature(aircraft_folder_path) 
			train_flight_data.append(data)
			train_flight_label.append(label)

		test_flight_data = []
		test_flight_label = []
		for aircraft_num in test_aircraft_num:
			aircraft_folder_path = csv_folder_path+'/Aircraft '+str(aircraft_num)
			data, label = get_aircraft_feature(aircraft_folder_path)
			test_flight_data.append(data)
			test_flight_label.append(label)
		
		train_data = np.vstack(train_flight_data)
		train_label = np.hstack(train_flight_label)
		# train_label = train_label.reshape(train_label.shape[0],)

		test_data = np.vstack(test_flight_data)
		test_label = np.hstack(test_flight_label)
		# test_label = test_label.reshape(test_label.shape[0],)

		np.save(train_test_data_folder+"/train_data.npy", train_data)
		np.save(train_test_data_folder+"/train_label.npy", train_label)
		np.save(train_test_data_folder+"/test_data.npy", test_data)
		np.save(train_test_data_folder+"/test_label.npy", test_label)
	else:
		train_data = np.load(train_test_data_folder+"/train_data.npy")
		train_label = np.load(train_test_data_folder+"/train_label.npy")
		test_data = np.load(train_test_data_folder+"/test_data.npy")
		test_label = np.load(train_test_data_folder+"/test_label.npy")

	return train_data, train_label, test_data, test_label

#if __name__ == "__main__":
	#cross_validation_on_DA_features([1,2,3,4], [5])
	# cross_validation_on_image([1,2,3,5], [4])
	# cross_validation_on_image([1,2,4,5], [3])
	# cross_validation_on_image([1,3,4,5], [2])
	# cross_validation_on_image([2,3,4,5], [1])