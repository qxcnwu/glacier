import tensorflow as tf
import os
import random
from tqdm import tqdm
import argparse
import sys

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_record(jpgPath:os.path,labelPath:os.path,trainPath:os.path,testPath:os.path,splitNum:int):
    """
    制作tfrecord文件
    :param jpgPath: 图像文件夹
    :param labelPath: 标签文件夹
    :param trainPath: 训练record文件保存位置
    :param testPath: 测试record文件保存位置
    :param splitNum: 分割值
    :return:
    """
    files=os.listdir(jpgPath)
    random.shuffle(files)
    split_num=int(len(files)*float(splitNum))
    train_list = files[:split_num]
    test_list = files[split_num:-1]
	
    with tf.io.TFRecordWriter(trainPath) as writer:
        for file in tqdm(train_list):
            jpg_path=os.path.join(jpgPath,file)
            label_path=os.path.join(labelPath,file)
            with tf.compat.v1.gfile.GFile(jpg_path, 'rb') as fid:
                jpg = fid.read()
            with tf.compat.v1.gfile.GFile(label_path, 'rb') as fid:
                label = fid.read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'jpg': _bytes_feature(jpg),
                }))
            writer.write(example.SerializeToString())
    writer.close()

    with tf.io.TFRecordWriter(testPath) as writer:
        for file in tqdm(test_list):
            jpg_path=os.path.join(jpgPath,file)
            label_path=os.path.join(labelPath,file)
            with tf.compat.v1.gfile.GFile(jpg_path, 'rb') as fid:
                jpg = fid.read()
            with tf.compat.v1.gfile.GFile(label_path, 'rb') as fid:
                label = fid.read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'jpg': _bytes_feature(jpg),
                }))
            writer.write(example.SerializeToString())
    writer.close()
    return 0

def main(argv):
    """
    # 读取系统输入
    :param argv: argv
    :return: 执行函数main
    """
    parser = argparse.ArgumentParser(prog=argv[0], description="DOWNLOAD WITH THREADING")
    parser.add_argument('-j', '--jpgPath', dest='jpgPath', metavar='DIR', help='JPG FILES SAVE DIR', required=True)
    parser.add_argument('-l', '--labelPath', dest='labelPath', metavar='DIR',help='LABEL FILES SAVE DIR', required=True)
    parser.add_argument('-t1', '--trainRecord', dest='trainRecord', metavar='PATH', help='TRAIN RECORD SAVE PATH',required=True)
    parser.add_argument('-t2', '--testRecord', dest='testRecord', metavar='PATH', help='TEST RECORD SAVE PATH',required=True)
    parser.add_argument('-n', '--splitNum', dest='splitNum', metavar='NUM', help='TRAIN||TEST NUMBER', required=True)
    args = parser.parse_args(argv[1:])
    return make_record(args.jpgPath,args.labelPath,args.trainRecord,args.testRecord,args.splitNum)

if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)