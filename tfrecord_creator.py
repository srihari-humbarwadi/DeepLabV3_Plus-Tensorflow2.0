import tensorflow as tf
from PIL import Image
import numpy as np

import glob, os, tqdm


_NUM_SHARDS = 4


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# Create a dictionary with features that may be relevant.
def image_example(image_string, mask_string, image_shape, mask_shape):
  #image_shape = tf.image.decode_jpeg(image_string).shape
  #mask_shape = tf.image.decode_png(mask_string).shape

  feature = {
      'image/height': _int64_feature(image_shape[0]),
      'image/width': _int64_feature(image_shape[1]),
      'image/depth': _int64_feature(image_shape[2]),
      'image_raw': _bytes_feature(image_string),
      'mask/height': _int64_feature(mask_shape[0]),
      'mask/width': _int64_feature(mask_shape[1]),
      'mask/depth': _int64_feature(1),
      'mask_raw': _bytes_feature(mask_string)
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords(image_dir,mask_dir,out_dir, training_data=True):
    '''Args:
    image_paths : List of file-paths for the images
    labels : class-labels for images(vector of size len(image_paths)X1)
    out_path : Destination of TFRecords output file
    size : resize dimensions
    '''

    assert os.path.exists(out_dir), 'directory doesnot exist :: {}'.format(out_dir)
   
    image_paths = glob.glob(os.path.join(image_dir,'*.jpg'))
    num_images = len(image_paths)
    #print(num_images)
    unique_vals=[]

    images_per_shard = num_images//_NUM_SHARDS

    shard_meta = {}
    start_index = 0

    for idx in range(_NUM_SHARDS):
      prefix = 'train' if training_data else 'test'
      end_index = min(start_index+images_per_shard, num_images)
      shard_meta['%s-%05d-of-%05d.tfrecord'%(prefix, idx+1, _NUM_SHARDS)] = (start_index, end_index)
      start_index = end_index+1

    # already_seen = [] 

    for key, (START, END) in shard_meta.items():

        print('Writing :: {}'.format(key))

        with tf.io.TFRecordWriter(os.path.join(out_dir, key)) as writer :

           for image_file in tqdm.tqdm(image_paths[START:END]):
                #img = cv2.imread(path)  # -1 to read as default dat type of original image and not as uint8
                # if image_file not in already_seen:
                #   print(image_file)
                #   already_seen.append(image_file)
                # else:
                #   raise ValueError('Repeated image {}'.format(image_file))
                with open(image_file, 'rb') as imf:
                  image_string = imf.read()
                image_array = np.array(Image.open(image_file))
                image_shape = image_array.shape
                if len(image_shape) !=3:
                  print('ignoring {}, shape : {}'.format(image_file, image_shape))
                  continue

                assert image_shape[2] == 3, 'expected image to have 3 channels but got {} instead'.format(image_shape[2])

                mask_file = image_file.replace(image_dir, mask_dir).replace('.jpg','.png')

                if os.path.isfile(mask_file)==False :# image has a mask
                  #create an all black mask
                  black_mask = Image.new(mode='L', size=(image_shape[1],image_shape[0]), color=0)
                  #save it as temp.png
                  black_mask.save('temp.png')
                  mask_file = 'temp.png'

                with open(mask_file, 'rb') as mmf:
                  mask_string = mmf.read()
                mask_array = np.array(Image.open(mask_file))
                unique_pixels = np.unique(mask_array).tolist()
                unique_vals = list(set(unique_vals).union(unique_pixels))
                mask_shape = mask_array.shape

                assert len(mask_shape) == 2, 'expected mask to have 1 channel but got {} instead'.format(mask_shape)

                assert mask_shape[0]==image_shape[0], 'mask and image height mismatch'
                assert mask_shape[1]==image_shape[1], 'mask and image width mismatch'

                tf_example = image_example(
                  image_string=image_string, 
                  mask_string=mask_string,
                  image_shape=image_shape,
                  mask_shape=mask_shape)

                writer.write(tf_example.SerializeToString())

    print('Num unique labels :: {}'.format(len(unique_vals)))



