import tensorflow as tf
import glob, os, tqdm


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
        'mask_raw': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

        image_string = parsed_example['image_raw']
        mask_string = parsed_example['mask_raw']
        #depth_string = parsed_example['depth_raw']

        # decode the raw bytes so it becomes a tensor with type

        image = tf.cast(tf.image.decode_jpeg(image_string), tf.uint8)
        image = tf.image.resize(image,(height, width))
        image = tf.cast(image, tf.keras.backend.floatx())
        image.set_shape([height, width,3])

        mask = tf.cast(tf.image.decode_png(mask_string), tf.uint8)
        mask = tf.image.resize(mask,(height, width), method='nearest')
        mask.set_shape([height, width,1])

        # mask = tf.cast(mask, tf.int32)
        normalized_image = image-[103.939, 116.779, 123.68]

        return normalized_image , mask
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)

    dataset = dataset.shuffle(buffer_size=4)

    dataset = dataset.repeat(-1) # Repeat the dataset this time
    dataset = dataset.batch(batch_size)    # Batch Size
    batch_dataset = dataset.prefetch(buffer_size=4)

    #iterator = batch_dataset.make_one_shot_iterator()   # Make an iterator
    #batch_features,batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels
    
    #return batch_features, batch_labels
    return batch_dataset
