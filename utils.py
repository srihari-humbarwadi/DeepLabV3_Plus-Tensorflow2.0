import tensorflow as tf
import tempfile, os, zipfile
# tempfile.gettempdir()
def get_miniade20k(dataset_path=os.getcwd()):
    num_classes = 152
    dataset_size = 2000
    zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/segmentation/mini_ADE20K.zip'
    path_to_zip_file = tf.keras.utils.get_file(
        'mini_ADE20K.zip',
        zip_url,
        # cache_dir=dataset_path, 
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'mini_ADE20K','images')
    annotation_dir = os.path.join(dataset_path, 'mini_ADE20K','annotations')

    return images_dir, annotation_dir, num_classes, dataset_size
