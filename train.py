import tensorflow as tf
import numpy as np
import os

from data_helpers import maybe_download_and_extract, load_training_data, load_test_data, load_class_names
import utilities.net as net
import train_evaluation


tf.app.flags.DEFINE_string(
	'train_dir', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
	'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
	'checkpoint_dir',
	os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
	'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint', 10,
							'Maximum number of checkpoints that TensorFlow will keep.')

tf.app.flags.DEFINE_integer('num_classes', 10,
							'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size', int(np.power(2, 9)),
							'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_epochs', 1,
							'Number of epochs for training.')

# Learning rate flags 

tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
	'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
	'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

# status flags

tf.app.flags.DEFINE_boolean('is_training', False,
							'Training/Testing.')

tf.app.flags.DEFINE_boolean('fine_tuning', False,
							'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('online_test', True,
							'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True,
							'Automatically put the variables on CPU if there is no GPU support.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
							'Demonstrate which variables are on what device.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

if not os.path.isabs(FLAGS.train_dir):
	raise ValueError('You must assign absolute path for --train_dir')

if not os.path.isabs(FLAGS.checkpoint_dir):
	raise ValueError('You must assign absolute path for --checkpoint_dir')



maybe_download_and_extract()
images_train, cls_train, labels_train = load_training_data()
images_test, cls_test, labels_test = load_test_data()
tensors_key = ['images_train', 'labels_train', 'images_test', 'labels_test']
tensors = [images_train, labels_train, images_test, labels_test]
data = dict(zip(tensors_key, tensors))

num_train_samples = images_train.shape[0]
height = 32
width = 32
num_channels = 3
print(load_class_names())


graph = tf.Graph()
with graph.as_default():

	global_step = tf.Variable(0, name="global_step", trainable=False)

	decay_steps = int(num_train_samples / FLAGS.batch_size *
					  FLAGS.num_epochs_per_decay)
	learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
											   global_step,
											   decay_steps,
											   FLAGS.learning_rate_decay_factor,
											   staircase=True,
											   name='exponential_decay_learning_rate')


	image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
	label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
	dropout_param = tf.placeholder(tf.float32)


	arg_scope = net.net_arg_scope(weight_decay=0.0005, is_training=FLAGS.is_training)
	with tf.contrib.framework.arg_scope(arg_scope):
		logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes,
												  dropout_keep_prob=dropout_param,
												  is_training=FLAGS.is_training)

	# Define loss
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))

	# Accuracy
	with tf.name_scope('accuracy'):
		# Evaluate the model
		correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))

		# Accuracy calculation
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

	with tf.name_scope('train'):
		grads_and_vars = optimizer.compute_gradients(loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


	arr = np.random.randint(low=0, high=images_train.shape[0], size=(3,))
	tf.summary.image('images', images_train[arr], max_outputs=3,
					 collections=['per_epoch_train'])


	for end_point in end_points:
		x = end_points[end_point]
		tf.summary.scalar('sparsity/' + end_point,
						  tf.nn.zero_fraction(x), collections=['train', 'test'])
		tf.summary.histogram('activations/' + end_point, x, collections=['per_epoch_train'])

	# Summaries for loss and accuracy
	tf.summary.scalar("loss", loss, collections=['train', 'test'])
	tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
	tf.summary.scalar("global_step", global_step, collections=['train'])
	tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

	# Merge all summaries together.
	summary_train_op = tf.summary.merge_all('train')
	summary_test_op = tf.summary.merge_all('test')
	summary_epoch_train_op = tf.summary.merge_all('per_epoch_train')

	tensors_key = ['cost', 'accuracy', 'train_op', 'global_step', 'image_place', 'label_place', 'dropout_param',
				   'summary_train_op', 'summary_test_op', 'summary_epoch_train_op']
	tensors = [loss, accuracy, train_op, global_step, image_place, label_place, dropout_param, summary_train_op,
			   summary_test_op, summary_epoch_train_op]
	tensors_dictionary = dict(zip(tensors_key, tensors))

	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(graph=graph, config=session_conf)

	with sess.as_default():
		# Run the saver.
		# 'max_to_keep' flag determines the maximum number of models that the tensorflow save and keep. default by TensorFlow = 5.
		saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)

		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		train_evaluation.train(sess=sess, saver=saver, tensors=tensors_dictionary, data=data,
							   train_dir=FLAGS.train_dir,
							   finetuning=FLAGS.fine_tuning, online_test=FLAGS.online_test,
							   num_epochs=FLAGS.num_epochs, checkpoint_dir=FLAGS.checkpoint_dir,
							   batch_size=FLAGS.batch_size)

		# Test in the end of experiment.
		train_evaluation.evaluation(sess=sess, saver=saver, tensors=tensors_dictionary, data=data,
									checkpoint_dir=FLAGS.checkpoint_dir)



