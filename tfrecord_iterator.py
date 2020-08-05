import tensorflow as tf
import glob, os, tqdm, cv2
import numpy as np

def pad_resize(image, height, width, channels=3):
    """Summary
    
    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description
    
    Returns:
        numpy nd.array: Description
    """

    image = image.astype(np.uint8)

    padded_image = np.zeros(shape=(height.astype(int), width.astype(int),channels), dtype=np.uint8)
    h,w,_ =  image.shape
    padded_image[:h,:w,:] = image
    return padded_image

@tf.function
def decode_pad_img(image_string, pad_height, pad_width):
  """Summary
  
  Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      scale (TYPE): Description
  
  Returns:
      tf.tensor: Description
  """
  image = tf.image.decode_jpeg(image_string)
  image = tf.numpy_function(pad_resize, [image, pad_height, pad_width], Tout=tf.uint8)
  image = tf.cast(image, tf.keras.backend.floatx())
  #image.set_shape([None, None, 3])
  return image-[103.939, 116.779, 123.68]

@tf.function
def decode_pad_msk(mask_string, pad_height, pad_width):
  """Summary
  
  Args:
      mask_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      scale (TYPE): Description
  
  Returns:
      tf.tensor: Description
  """
  mask = tf.image.decode_png(mask_string)
  mask = tf.numpy_function(pad_resize, [mask, pad_height, pad_width, 1], Tout=tf.uint8)
  #mask.set_shape([None, None, 3])
  return mask

def parse_tfrecords(filenames, height, width, batch_size=32):

    def _parse_function(serialized):

        features = {
                'image/height': tf.io.FixedLenFeature([], tf.int64),
                'image/width': tf.io.FixedLenFeature([], tf.int64),
                'image/depth': tf.io.FixedLenFeature([], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'mask/height': tf.io.FixedLenFeature([], tf.int64),
                'mask/width': tf.io.FixedLenFeature([], tf.int64),
                'mask/depth': tf.io.FixedLenFeature([], tf.int64),
                'mask_raw': tf.io.FixedLenFeature([], tf.string)}

        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        image_batch = tf.map_fn(lambda x: decode_pad_img(x, max_height, max_width), parsed_example['image_raw'], dtype=tf.keras.backend.floatx())
        image_batch.set_shape([None, None, None,3])

        mask_batch = tf.map_fn(lambda x: decode_pad_msk(x, max_height, max_width), parsed_example['mask_raw'], dtype=tf.uint8)
        mask_batch.set_shape([None, None, None,1])


        # parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
        # image_string = parsed_example['image_raw']
        # mask_string = parsed_example['mask_raw']
        #depth_string = parsed_example['depth_raw']

        # decode the raw bytes so it becomes a tensor with type

        # image = tf.cast(tf.image.decode_jpeg(image_string), tf.uint8)
        image_batch = tf.image.resize(image_batch,(height, width))
        image_batch = tf.cast(image_batch, tf.keras.backend.floatx())
        # image_batch.set_shape([None, height, width,3])

        # mask = tf.cast(tf.image.decode_png(mask_string), tf.uint8)
        mask_batch = tf.image.resize(mask_batch,(height, width), method='nearest')
        # mask_batch.set_shape([None, height, width,1])

        # mask = tf.cast(mask, tf.int32)
        # normalized_image = image-[103.939, 116.779, 123.68]

        return image_batch, mask_batch
    
    # dataset = tf.data.TFRecordDataset(filenames=filenames)
    # dataset = dataset.map(_parse_function, num_parallel_calls=4)

    # dataset = dataset.shuffle(buffer_size=4)

    # dataset = dataset.repeat(-1) # Repeat the dataset this time
    # dataset = dataset.batch(batch_size)    # Batch Size
    # batch_dataset = dataset.prefetch(buffer_size=4)

    filenames=tf.io.gfile.glob(filenames)
    dataset=tf.data.Dataset.from_tensor_slices(filenames).shuffle(buffer_size=16).repeat(-1)

    #dataset = tf.data.Dataset.list_files(filenames).repeat(-1)
    dataset = dataset.interleave(
      tf.data.TFRecordDataset, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      cycle_length=4, 
      block_length=16)

    dataset = dataset.batch(
      batch_size, 
      drop_remainder=True)    # Batch Size

    dataset = dataset.map(
      _parse_function, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # dataset = dataset.shuffle(buffer_size=256)

    # dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
    # return batch_dataset
