#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from deeplab import DeepLabV3Plus

print('TensorFlow', tf.__version__)


# In[ ]:


batch_size = 24
H, W = 384, 384
num_classes = 34

image_list = sorted(glob('cityscapes/dataset/train_images/*'))
mask_list = sorted(glob('cityscapes/dataset/train_masks/*'))

val_image_list = sorted(glob('cityscapes/dataset/val_images/*'))
val_mask_list = sorted(glob('cityscapes/dataset/val_masks/*'))

print('Found', len(image_list), 'training images')
print('Found', len(val_image_list), 'validation images')

for i in range(len(image_list)):
    assert image_list[i].split('/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
    
for i in range(len(val_image_list)):
    assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[0] == val_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]


# In[ ]:


id_to_color = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0), 4: (0, 0, 0), 5: (111, 74, 0), 6: (81, 0, 81), 7: (128, 64, 128), 8: (244, 35, 232), 9: (250, 170, 160), 10: (230, 150, 140), 11: (70, 70, 70), 12: (102, 102, 156), 13: (190, 153, 153), 14: (180, 165, 180), 15: (150, 100, 100), 16: (150, 120, 90), 17: (153, 153, 153), 18: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60, 100), 29: (0, 0, 90), 30: (0, 0, 110), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32), -1: (0, 0, 142)}


# In[ ]:


def get_image(image_path, img_height=512, img_width=1024, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        img = tf.image.random_brightness(img, max_delta=50.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.case([
            (tf.greater(flip , 0), lambda : tf.image.flip_left_right(img))
            ], default=lambda : img)
        img  = img[:,:,::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[img_height, img_width]), dtype=tf.uint8)
        img = tf.case([
            (tf.greater(flip , 0), lambda : tf.image.flip_left_right(img))
            ], default=lambda : img)
    return img

def random_crop(image, mask, H=384, W=384):
    image_dims = image.shape
    offset_h = tf.random.uniform(shape=(1,), maxval=image_dims[0]-H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(shape=(1,), maxval=image_dims[1]-W, dtype=tf.int32)[0]
    
    image = tf.image.crop_to_bounding_box(image, 
                                          offset_height=offset_h, 
                                          offset_width=offset_w, 
                                          target_height=H, 
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask, 
                                          offset_height=offset_h, 
                                          offset_width=offset_w, 
                                          target_height=H, 
                                          target_width=W)
    return image, mask

def load_data(image_path, mask_path, H=384, W=384):
    flip = tf.random.uniform(shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(mask_path, mask=True, flip=flip)
    image, mask = random_crop(image, mask, H=H, W=W)
    mask = tf.one_hot(tf.squeeze(mask), depth=num_classes)
    return image, mask


# In[ ]:


train_dataset = tf.data.Dataset.from_tensor_slices((image_list, 
                                                    mask_list))
train_dataset = train_dataset.shuffle(buffer_size=128)
train_dataset = train_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data, 
                                       batch_size=batch_size, 
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE, 
                                       drop_remainder=True))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(train_dataset)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, 
                                                  val_mask_list))
val_dataset = val_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data, 
                                       batch_size=batch_size, 
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE, 
                                       drop_remainder=True))
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeepLabV3Plus(H, W, num_classes)
    model.compile(loss=tf.losses.categorical_crossentropy, 
                  optimizer=tf.optimizers.Adam(learning_rate=5e-4), 
                  metrics=['accuracy'])


# In[ ]:


tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)
callbacks = [mc, tb]


# In[ ]:


model.fit(train_dataset,
          steps_per_epoch=len(image_list)//batch_size,
          epochs=100,
          validation_data=val_dataset,
          validation_steps=len(val_image_list)//batch_size, 
          callbacks=callbacks)

