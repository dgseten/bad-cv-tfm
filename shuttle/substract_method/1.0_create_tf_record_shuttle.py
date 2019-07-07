from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

import skimage
from matplotlib import pyplot as plt
import cv2
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('labeled_path', '/media/seten/Datos/diego/TFM/dataset_tfm/shuttlecock/secuencias', 'Path where labeled images are')
flags.DEFINE_string('output_path', '/home/seten/TFM/data/train_shuttlecock_subtract.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '/home/seten/TFM/data/shuttle_map.pbtxt','Path to label map proto')
FLAGS = flags.FLAGS


def find_labeled_images(original_corpus_path):

    print("Finding labeled images")
    images = []
    xmls=[]
    # recorremos las minisecuencias
    for folder in os.listdir(original_corpus_path):
        seq_folder = os.path.join(original_corpus_path,folder)
        if os.path.isdir(seq_folder):
            print("Adding sequence {}".format(seq_folder))
            # recorremos la minisecuencia
            sorted_ims_in_dir = [i for i in sorted(filter(lambda a: a.endswith(".jpeg"),os.listdir(seq_folder)))]
            for i in range(len(sorted_ims_in_dir)):
                im = sorted_ims_in_dir[i]
                xml_file = os.path.join(seq_folder,im.replace(".jpeg",".xml"))
                if os.path.isfile(xml_file):
                    xmls.append(xml_file)
                    im_current = os.path.join(seq_folder, im)
                    im_prev = os.path.join(seq_folder,sorted_ims_in_dir[i-1]) if i>0 else im_current
                    ims_tuple = (im_current,im_prev)
                    images.append(ims_tuple)

    return images,xmls



def create_tf_example(img_tuple, xml_path, label_map_dict, ignore_difficult_instances=False):

    with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)
    data = data['annotation']



    current_frame = skimage.io.imread(img_tuple[0])
    prev_frame = skimage.io.imread(img_tuple[1])
    image_s = cv2.subtract(current_frame,prev_frame)
    #retval, encoded_jpg = cv2.imencode('.jpg', image_s)
    Image.fromarray(image_s).save("tmp.jpeg")

    with tf.gfile.GFile("tmp.jpeg", 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    logging.info("Starting TF Record conversor ...")
    logging.info("Reading dataset from: {}".format(FLAGS.labeled_path))
    logging.info("Output TF Record in: {}".format(FLAGS.output_path))
    logging.info("Using label map file: {}".format(FLAGS.output_path))

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    images, xmls = find_labeled_images(FLAGS.labeled_path)

    print("Storing in tf record {} images".format(len(xmls)))
    i = 0
    for image, label_img_xml in zip(images, xmls):
        tf_example = create_tf_example(image, label_img_xml,label_map_dict)
        writer.write(tf_example.SerializeToString())
        i+=1
        print("{}/{}".format(i,len(xmls)))

    writer.close()


if __name__ == '__main__':
    tf.app.run()
