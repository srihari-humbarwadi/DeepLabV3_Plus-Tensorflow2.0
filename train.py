from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from deeplab import DeepLabV3Plus

import os

from tfrecord_iterator import parse_tfrecords
from tfrecord_creator import create_tfrecords
from utils import get_miniade20k

print('TensorFlow', tf.__version__)

images_path, xml_path, num_classes, dataset_size = get_miniade20k()

batch_size = 2
H, W = 512, 512

tfrecord_dir = os.path.join(os.getcwd(), 'tfrecords')
os.makedirs(tfrecord_dir, exist_ok=True)

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

create_tfrecords(images_path, xml_path, tfrecord_dir)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
model = DeepLabV3Plus(H, W, num_classes)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.9997
        layer.epsilon = 1e-5
    elif isinstance(layer, tf.keras.layers.Conv2D):
        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
model.compile(loss=loss, 
              optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
              metrics=['accuracy'])


tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'top_weights_{epoch:02d}.h5'),
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True',
                     period=5, 
                     verbose=1)
callbacks = [mc, tb]

train_tfrecords = os.path.join(tfrecord_dir, 'train*.tfrecord')
input_function = parse_tfrecords(
    filenames=train_tfrecords,
    height=H,
    width=W,
    batch_size=batch_size)

# model.fit(input_function,
#           steps_per_epoch=dataset_size//batch_size,
#           epochs=300,
#           validation_data=input_function,
#           validation_steps=dataset_size//batch_size,
#           callbacks=callbacks)

import cv2
import numpy as np

viz_data_path = os.path.join(os.getcwd(), 'viz')
os.makedirs(viz_data_path, exist_ok=True)

i=0

for image, mask in input_function.take(10):
  for index in range(batch_size):
    im = image[index].numpy()
    msk = mask[index].numpy()
    im = im+[103.939, 116.779, 123.68]

    cv2.imwrite(os.path.join(viz_data_path, '{}_image.jpg'.format(i)), im.astype(np.uint8))
    cv2.imwrite(os.path.join(viz_data_path, '{}_mask.png'.format(i)), msk.astype(np.uint8))

    i = i+1
#     # print(image[index].shape, mask[index].shape)
