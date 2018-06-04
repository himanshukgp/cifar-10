import numpy as np
import pickle
import os
import sys
import urllib.request
import tarfile
import zipfile


data_path = "data/CIFAR-10/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file


def print_progress(progress, epoch_num, loss):
	barLength = 30
	assert type(progress) is float, "id is not a float: %r" % id
	assert 0 <= progress <= 1, "variable should be between zero and one!"
	status = ""
	if progress >= 1:
		progress = 1
		status = "\r\n"
	indicator = int(round(barLength*progress))
	list = [str(epoch_num), "#"*indicator , "-"*(barLength-indicator), progress*100, loss, status]
	text = "\rEpoch {0[0]} {0[1]} {0[2]} %{0[3]:.2f} loss={0[4]:.3f} {0[5]}".format(list)
	sys.stdout.write(text)
	sys.stdout.flush()


def one_hot_encoded(class_numbers, num_classes=None):
	if num_classes is None:
		num_classes = np.max(class_numbers) + 1

	return np.eye(num_classes, dtype=float)[class_numbers]

def _unpickle(filename):
	file_path = os.path.join(data_path, "cifar-10-batches-py/", filename)
	print("Loading data: " + file_path)
	with open(file_path, mode='rb') as file:
		data = pickle.load(file, encoding='bytes')
	return data

def _convert_images(raw):
	raw_float = np.array(raw, dtype=float) / 255.0
	images = raw_float.reshape([-1, num_channels, img_size, img_size])
	images = images.transpose([0, 2, 3, 1])
	return images

def _load_data(filename):
	data = _unpickle(filename)
	raw_images = data[b'data']
	cls = np.array(data[b'labels'])
	images = _convert_images(raw_images)
	return images, cls


def _print_download_progress(count, block_size, total_size):
	pct_complete = float(count * block_size) / total_size
	msg = "\r- Download progress: {0:.1%}".format(pct_complete)
	sys.stdout.write(msg)
	sys.stdout.flush()

def maybe_download_and_extract(url=data_url, download_dir=data_path):
	filename = url.split('/')[-1]
	file_path = os.path.join(download_dir, filename)
	
	if not os.path.exists(file_path):
		if not os.path.exists(download_dir):
			os.makedirs(download_dir)

		file_path, _ = urllib.request.urlretrieve(url=url,
												  filename=file_path, 
												  reporthook=_print_download_progress)
		print()
		print("Download finished. Extracting files.")

		if file_path.endswith(".zip"):
			# Unpack the zip-file.
			zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
		elif file_path.endswith((".tar.gz", ".tgz")):
			# Unpack the tar-ball.
			tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

		print("Done.")
	else:
		print("Data has apparently already been downloaded and unpacked.")

def load_class_names():
	raw = _unpickle(filename="batches.meta")[b'label_names']
	names = [x.decode('utf-8') for x in raw]
	return names

def load_training_data():
	images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
	cls = np.zeros(shape=[_num_images_train], dtype=int)
	begin = 0

	for i in range(_num_files_train):
		images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
		num_images = len(images_batch)
		end = begin + num_images
		images[begin:end, :] = images_batch
		cls[begin:end] = cls_batch
		begin = end

	return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
	images, cls = _load_data(filename="test_batch")
	return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)



