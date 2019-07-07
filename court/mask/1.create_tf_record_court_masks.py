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
flags.DEFINE_string('labeled_path', '/media/seten/Datos/diego/TFM/dataset_tfm/court_poles', 'Path where labeled images are')
flags.DEFINE_string('output_path', '/home/seten/TFM/data/train_court_masks.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '/home/seten/TFM/data/court_map.pbtxt','Path to label map proto')
FLAGS = flags.FLAGS


def find_labeled_images(original_corpus_path):
    jsons = []
    for file in filter(lambda a: a.endswith(".json"), os.listdir(original_corpus_path)):
        jsons.append(os.path.join(original_corpus_path, file))
    images = [a for a in map(lambda f: f.replace(".json", ".jpeg"), jsons)]
    return images, jsons


def create_tf_example(img_path, json_path, label_map_dict, ignore_difficult_instances=False):

    with tf.gfile.GFile(json_path, 'r') as fid:
        json_str = fid.read()
    data = json.loads(json_str)
    #data = dataset_util.recursive_parse_xml_to_dict(xml)


    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width,height = image.size


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
    if len(data["shapes"])>1:
        print("problem img {}".format(data["imagePath"]))

    for obj in data['shapes']:
        difficult = False
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        for p in obj["points"]:
            if p[0] >= width:
                print("problem {} and img size {} and img {}".format(p,image.size,data["imagePath"]))
                p[0] = width-1
            if p[0] < 0:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[0] = 0
            if p[1] >= height:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[1] = height - 1
            if p[1] < 0:
                print("problem {} and img size {} and img {}".format(p, image.size, data["imagePath"]))
                p[1] = 0


        s_xmin = min([p[0] for p in obj["points"]])
        s_ymin = min([p[1] for p in obj["points"]])
        s_xmax = max([p[0] for p in obj["points"]])
        s_ymax = max([p[1] for p in obj["points"]])

        xmin.append(max(float(s_xmin) / width,0))
        ymin.append(max(float(s_ymin) / height,0))
        xmax.append(min(float(s_xmax) / width,1.0))
        ymax.append(min(float(s_ymax) / height,1.0))
        classes_text.append(obj["label"].encode('utf8'))
        classes.append(label_map_dict[obj["label"]])
        truncated.append(int(0))
        poses.append("Unspecified".encode('utf8'))

        # create masks
        polygon = [tuple(p) for p in obj["points"]]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = numpy.array(img)
        masks.append(mask)
        #img.save("mask.png")

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

    encoded_mask_png_list = []
    for mask in masks:
        img = Image.fromarray(mask)
        output = io.BytesIO()
        img.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def main(_):
    logging.info("Starting TF Record conversor with masks...")
    logging.info("Reading dataset from: {}".format(FLAGS.labeled_path))
    logging.info("Output TF Record in: {}".format(FLAGS.output_path))
    logging.info("Using label map file: {}".format(FLAGS.output_path))

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    images, jsons = find_labeled_images(FLAGS.labeled_path)

    for image, label_img_json in zip(images, jsons):
        tf_example = create_tf_example(image, label_img_json,label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
