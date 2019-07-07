from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import json

import tensorflow as tf

import numpy
from PIL import Image, ImageDraw

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('labeled_path', '/media/seten/Datos/diego/TFM/dataset_tfm/court/train',
                    'Path where labeled images are')
flags.DEFINE_string('output_path', '/home/seten/TFM/data/train_court_corners.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '/home/seten/TFM/data/court_corners_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('debug_images_path', '/home/seten/TFM/debug_images/court_corners_train',
                    'Path for store debug images folder')
FLAGS = flags.FLAGS
CORNER_BBOX_SIZE = 7

count_skip = 0

def find_labeled_images(original_corpus_path):
    xmls = []
    for file in filter(lambda a: a.endswith(".json"), os.listdir(original_corpus_path)):
        xmls.append(os.path.join(original_corpus_path, file))
    images = [a for a in map(lambda f: f.replace(".json", ".jpeg"), xmls)]
    return images, xmls


def create_tf_example(img_path, json_path, label_map_dict, ignore_difficult_instances=False):
    with tf.gfile.GFile(json_path, 'r') as fid:
        json_str = fid.read()
    data = json.loads(json_str)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    image_draw = ImageDraw.Draw(image, "RGBA")
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
    if len(data["shapes"]) > 1:
        print("problem img {}".format(data["imagePath"]))



    for obj in data['shapes']:
        difficult = False
        if ignore_difficult_instances and difficult:
            continue



        for p in obj["points"]:
            if p[0] >= width:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[0] = width - 1
            if p[0] < 0:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[0] = 0
            if p[1] >= height:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[1] = height - 1
            if p[1] < 0:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[1] = 0

        for p in obj["points"]:
            # For each point we have to build a artificial rectangle.

            s_xmin = p[0] - CORNER_BBOX_SIZE
            s_ymin = p[1] - CORNER_BBOX_SIZE
            s_xmax = p[0] + CORNER_BBOX_SIZE
            s_ymax = p[1] + CORNER_BBOX_SIZE

            if s_xmax >= width:
                s_xmax = width - 1
            if s_xmin < 0:
                s_xmin = 0
            if s_ymax >= height:
                s_ymax = height - 1
            if s_ymin < 0:
                s_ymin = 0

            xmin.append(max(float(s_xmin) / width, 0))
            ymin.append(max(float(s_ymin) / height, 0))
            xmax.append(min(float(s_xmax) / width, 1.0))
            ymax.append(min(float(s_ymax) / height, 1.0))
            classes_text.append("corner".encode('utf8'))
            classes.append(label_map_dict["corner"])
            truncated.append(int(0))
            poses.append("Unspecified".encode('utf8'))
            difficult_obj.append(int(difficult))

            # TODO debug image
            image_draw.rectangle(((s_xmin, s_ymin), (s_xmax, s_ymax)), fill="black")
            os.makedirs(FLAGS.debug_images_path, exist_ok=True)
            debug_out_path = os.path.join(FLAGS.debug_images_path, data["imagePath"])
            image.save(debug_out_path)

    global count_skip
    count_skip +=1

    if count_skip == 25:
        print("sssss")
        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['imagePath'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['imagePath'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature([]),
            'image/object/bbox/xmax': dataset_util.float_list_feature([]),
            'image/object/bbox/ymin': dataset_util.float_list_feature([]),
            'image/object/bbox/ymax': dataset_util.float_list_feature([]),
            'image/object/class/text': dataset_util.bytes_list_feature([]),
            'image/object/class/label': dataset_util.int64_list_feature([]),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }
    else:
        feature_dict = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['imagePath'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['imagePath'].encode('utf8')),
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
        }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def main(_):
    logging.info("Starting TF Record conversor with masks...")
    logging.info("Reading dataset from: {}".format(FLAGS.labeled_path))
    logging.info("Output TF Record in: {}".format(FLAGS.output_path))
    logging.info("Using label map file: {}".format(FLAGS.output_path))
    logging.warning("Using debug folder path: {}".format(FLAGS.debug_images_path))

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    images, jsons = find_labeled_images(FLAGS.labeled_path)

    for image, label_img_json in zip(images, jsons):
        tf_example = create_tf_example(image, label_img_json, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
