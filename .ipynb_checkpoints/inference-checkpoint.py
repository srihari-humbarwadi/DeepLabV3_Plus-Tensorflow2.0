import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from deeplab import DeepLabV3Plus
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
from glob import glob
import pickle
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils import Sequence
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tensorflow.keras.applications.resnet50 import preprocess_input

h, w = 800, 1600
with open('cityscapes_dict.pkl', 'rb') as f:
    id_to_color = pickle.load(f)['color_map']

model = DeepLabV3Plus(h, w, 34)
model.load_weights('top_weights.h5')


def pipeline(image, video=True, return_seg=False, fname='', folder=''):
    global b
    alpha = 0.5
    dims = image.shape
    image = cv2.resize(image, (w, h))
    x = image.copy()
    z = model.predict(preprocess_input(np.expand_dims(x, axis=0)))
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)

    img_color = image.copy()
    for i in np.unique(y):
        if i in id_to_color:
            img_color[y == i] = id_to_color[i]
    disp = img_color.copy()
    if video:
        cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
        return img_color
    if return_seg:
        return img_color / 255.
    else:
        cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
#         plt.figure(figsize=(20, 10))
#         out = np.concatenate([image/255, img_color/255, disp/255], axis=1)

#         plt.imshow(img_color/255.0)
#         plt.imshow(out)
        return cv2.imwrite(f'outputs/{folder}/{fname}', cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))


image_dir = '/home/mia/backup/research/autonomous_driving/cityscapes/dataset/val_images'
image_list = os.listdir(image_dir)
image_list.sort()
print(f'{len(image_list)} frames found')


test = load_img(f'{image_dir}/{image_list[1]}')
test = img_to_array(test)
pipeline(test, video=False)

for image_dir in ['stuttgart_00', 'stuttgart_01', 'stuttgart_02']:
    os.mkdir(f'outputs/{image_dir}')
    image_list = os.listdir(image_dir)
    image_list.sort()
    print(f'{len(image_list)} frames found')
    for i in tqdm(range(len(image_list))):
        try:
            test = load_img(f'{image_dir}/{image_list[i]}')
            test = img_to_array(test)
            segmap = pipeline(test, video=False,
                              fname=f'{image_list[i]}', folder=image_dir)
            if segmap == False:
                break
        except Exception as e:
            print(str(e))
    clip = ImageSequenceClip(
        sorted(glob(f'outputs/{image_dir}/*')), fps=18, load_images=True)
    clip.write_videofile(f'{image_dir}.mp4')
