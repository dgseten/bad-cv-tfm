from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import json
from pathlib import Path

import PIL.Image
import tensorflow as tf

import skimage
from matplotlib import pyplot as plt
import cv2
from PIL import Image, ImageDraw

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

LABELED_PATH = '/media/seten/Datos/diego/TFM/dataset_tfm/shuttlecock/secuencias/test'
OUTPUT_PATH = '/home/seten/TFM/debug_images/shuttlecock_memoria'

COURT_BY_SEQUENCE = {
    41: [(628, 441), (1286, 441), (1467, 993), (455, 993)],
    42: [(628, 441), (1286, 441), (1467, 993), (455, 993)],
    43: [(628, 441), (1286, 441), (1467, 993), (455, 993)],
    44: [(602, 576), (1319, 575), (1515, 1014), (403, 1016)],
    45: [(602, 576), (1319, 575), (1515, 1014), (403, 1016)],
}

POLES_BY_SEQUENCE = {
    41: [((561, 656), (557, 463)), ((1356, 657), (1361, 462))],
    42: [((561, 656), (557, 463)), ((1356, 657), (1361, 462))],
    43: [((561, 656), (557, 463)), ((1356, 657), (1361, 462))],
    44: [((524, 746), (521, 529)), ((1394, 746), (1394, 529))],
    45: [((524, 746), (521, 529)), ((1394, 746), (1394, 529))],
}


class ShuttleTrajectoryDrawer(object):

    def __init__(self, seq_id, img_width, img_height):
        self.seq_id = seq_id
        self.pil_image = Image.new('RGB', (img_width, img_height), color='white')
        self.draw = ImageDraw.Draw(self.pil_image)
        self._draw_court()
        self._draw_poles()

    def _draw_poles(self):
        # draw poles and net
        if int(self.seq_id) in POLES_BY_SEQUENCE:
            pl, pr = POLES_BY_SEQUENCE[int(self.seq_id)]
            self.draw.line(pl, fill="black", width=10)
            self.draw.line(pr, fill="black", width=10)
            self.draw.line((pl[1], pr[1]), fill="black", width=10)

    def _draw_court(self):
        if int(self.seq_id) in COURT_BY_SEQUENCE:
            tl, tr, br, bl = COURT_BY_SEQUENCE[int(self.seq_id)]
            self.draw.line((tl, tr), fill="black", width=10)
            self.draw.line((tl, bl), fill="black", width=10)
            self.draw.line((tr, br), fill="black", width=10)
            self.draw.line((br, bl), fill="black", width=10)

    def add_shuttle_position(self, xy: ((int, int), (int, int)),color="red"):
        if len(xy)>=4:
            draw_form = ((xy[0],xy[1]),(xy[2],xy[3]))
        else:
            draw_form = xy
        self.draw.rectangle((draw_form[0], draw_form[1]), fill=color)

    def draw_in_path(self, path):
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)

        self.pil_image.save(path)

    def close(self):
        self.pil_image.close()


def find_sequences_with_data(original_corpus_path):
    print("Finding labeled images")

    # recorremos las minisecuencias
    for folder in os.listdir(original_corpus_path):
        images = []
        jsons = []
        seq_folder = os.path.join(original_corpus_path, folder)
        if os.path.isdir(seq_folder):
            print("Adding sequence {}".format(seq_folder))
            # recorremos la minisecuencia
            sorted_ims_in_dir = [i for i in sorted(filter(lambda a: a.endswith(".jpeg"), os.listdir(seq_folder)))]
            for i in range(len(sorted_ims_in_dir)):
                im = sorted_ims_in_dir[i]
                json_file = os.path.join(seq_folder, im.replace(".jpeg", ".json"))
                if os.path.isfile(json_file):
                    jsons.append(json_file)
                    im_current = os.path.join(seq_folder, im)
                    images.append(im_current)

        yield folder, images, jsons


def main():
    # buscar video
    for sequence_name, images, jsons in find_sequences_with_data(LABELED_PATH):
        print(sequence_name, len(images), len(jsons))

        with open(jsons[0], "r") as file:
            json_str = file.read()
        data = json.loads(json_str)
        width = int(data['imageWidth'])
        height = int(data['imageHeight'])
        pil_image = Image.new('RGB', (width, height), color='white')
        # pil_image = Image.open(images[0])
        draw = ImageDraw.Draw(pil_image)

        # draw court
        if int(sequence_name) in COURT_BY_SEQUENCE:
            tl, tr, br, bl = COURT_BY_SEQUENCE[int(sequence_name)]
            draw.line((tl, tr), fill="black", width=10)
            draw.line((tl, bl), fill="black", width=10)
            draw.line((tr, br), fill="black", width=10)
            draw.line((br, bl), fill="black", width=10)

        # draw poles and net
        if int(sequence_name) in POLES_BY_SEQUENCE:
            pl, pr = POLES_BY_SEQUENCE[int(sequence_name)]
            draw.line(pl, fill="black", width=10)
            draw.line(pr, fill="black", width=10)
            draw.line((pl[1], pr[1]), fill="black", width=10)

        for json_path in jsons:
            with open(json_path, "r") as file:
                json_str = file.read()
            data = json.loads(json_str)
            for obj in data['shapes']:

                for p in obj["points"]:
                    if p[0] >= width:
                        p[0] = width - 1
                    if p[0] < 0:
                        p[0] = 0
                    if p[1] >= height:
                        p[1] = height - 1
                    if p[1] < 0:
                        p[1] = 0

                s_xmin = min([p[0] for p in obj["points"]])
                s_ymin = min([p[1] for p in obj["points"]])
                s_xmax = max([p[0] for p in obj["points"]])
                s_ymax = max([p[1] for p in obj["points"]])

                draw.rectangle(((s_xmin, s_ymin), (s_xmax, s_ymax)), fill="red")

        pil_image.save(Path(OUTPUT_PATH) / f'{sequence_name}_trayectory.png')
        pil_image.close()

        # pintar pista sobre la imagen

        # para cada anotacion pintar el rectángulo (o punto gordo, evaluar que queda mejor) donde está el volante
    """
    img = Image.new('RGB', (60, 30), color='red')
    img.save('pil_red.png')

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    

    print("Storing in tf record {} images".format(len(jsons)))
    i = 0
    for image, label_img_json in zip(images, jsons):
        tf_example = create_tf_example(image, label_img_json, label_map_dict)
        writer.write(tf_example.SerializeToString())
        i += 1
        print("{}/{}".format(i, len(jsons)))

    writer.close()
    """


if __name__ == '__main__':
    main()
