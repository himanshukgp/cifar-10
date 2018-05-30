import tensorflow as tf
import numpy as np
from data_helpers import maybe_download_and_extract, load_training_data, load_test_data, load_class_names


maybe_download_and_extract()
images_train, cls_train, labels_train = load_training_data()
images_test, cls_test, labels_test = load_test_data()
print(load_class_names())

