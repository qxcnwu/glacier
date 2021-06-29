#-*- coding : utf-8-*-

import tensorflow as tf

expected_features={
    'jpg':tf.io.FixedLenFeature((),tf.string,default_value=''),
    'ndvi':tf.io.FixedLenFeature((),tf.string,default_value=''),
    'label':tf.io.FixedLenFeature((),tf.string,default_value=''),
    'edge':tf.io.FixedLenFeature((),tf.string,default_value='')
}

def parse_example(serialized_example):
    features = tf.io.parse_single_example(serialized_example, expected_features)
    jpg=tf.io.decode_jpeg(features['jpg'])
    retyped_images = tf.cast(jpg, tf.float32)
    images = tf.reshape(retyped_images, [256, 256, 3])
    float_image = tf.image.per_image_standardization(images)
    float_image.set_shape([256, 256, 3])

    label = tf.io.decode_png(features['label'])
    retyped_label = tf.cast(label, tf.int32)
    label = tf.reshape(retyped_label, [256, 256, 1])
    label=label-1
    label.set_shape([256, 256, 1])

    return float_image,label

def tfrecords_reader_dataset(filenames,batch_size=32, n_parse_threads=5,shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename))
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example,num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

def make_list(num_class=10,val=0.01):
    import use_keras.TIFMODIFY as TF
    TF.make_label(num_class=num_class,valid_precent=val,tif_dir='C:/Users/86188/PycharmProjects/paddle/data/',label_file='C:/Users/86188/PycharmProjects/paddle/data/label.txt',
                  train_file='C:/Users/86188/PycharmProjects/paddle/data/train.txt',valid_file='C:/Users/86188/PycharmProjects/paddle/data/valid.txt')
    return 0

def make_record():
    import os
    from tqdm import tqdm
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    train_file=open('C:/Users/86188/PycharmProjects/paddle/data/train.txt')
    train_files=train_file.readlines()
    train_record='C:/Users/86188/PycharmProjects/paddle/data/train_128.record'
    with tf.io.TFRecordWriter(train_record) as writer:
        for file in tqdm(train_files):
            jpg_path=os.path.join('128/',file.split(' ')[0])
            ndvi_path=os.path.join('128/',file.split(' ')[1])
            label_path=os.path.join('128/',file.split(' ')[2].replace('\n',''))
            edge_path=os.path.join('boundary test/',file.split(' ')[2].replace('\n',''))
            with tf.compat.v1.gfile.GFile(jpg_path, 'rb') as fid:
                jpg = fid.read()
            with tf.compat.v1.gfile.GFile(label_path, 'rb') as fid:
                label = fid.read()
            with tf.compat.v1.gfile.GFile(edge_path, 'rb') as fid:
                edge = fid.read()
            with tf.compat.v1.gfile.GFile(ndvi_path, 'rb') as fid:
                ndvi = fid.read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'jpg': _bytes_feature(jpg),
                    'ndvi': _bytes_feature(ndvi),
                    'edge':_bytes_feature(edge)
                }))
            writer.write(example.SerializeToString())
    writer.close()
    print('TRAIN Writting End')
    val_file = open('C:/Users/86188/PycharmProjects/paddle/data/valid.txt')
    val_files = val_file.readlines()
    val_record = 'C:/Users/86188/PycharmProjects/paddle/data/valid_128.record'
    with tf.io.TFRecordWriter(val_record) as writer:
        for file in tqdm(val_files):
            jpg_path = os.path.join('128/', file.split(' ')[0])
            ndvi_path = os.path.join('128/', file.split(' ')[1])
            label_path = os.path.join('128/', file.split(' ')[2].replace('\n',''))
            edge_path = os.path.join('boundary test/', file.split(' ')[2].replace('\n', ''))
            with tf.compat.v1.gfile.GFile(jpg_path, 'rb') as fid:
                jpg = fid.read()
            with tf.compat.v1.gfile.GFile(label_path, 'rb') as fid:
                label = fid.read()
            with tf.compat.v1.gfile.GFile(ndvi_path, 'rb') as fid:
                ndvi = fid.read()
            with tf.compat.v1.gfile.GFile(edge_path, 'rb') as fid:
                edge = fid.read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'jpg': _bytes_feature(jpg),
                    'ndvi': _bytes_feature(ndvi),
                    'edge': _bytes_feature(edge)
                }))
            writer.write(example.SerializeToString())
    writer.close()
    print('VALID Writting End')
    return 0

train_record='C:/Users/86188/PycharmProjects/paddle/data/train_128.record'
val_record = 'C:/Users/86188/PycharmProjects/paddle/data/valid_128.record'

if __name__ == '__main__':
    pass
