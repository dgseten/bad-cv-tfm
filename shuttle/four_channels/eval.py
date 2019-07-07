from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

import skimage

from shuttle.eval.gen_eval_preccision_recall_model import generic_eval, find_sequences_with_data

import numpy as np


def build_image(sequence_per_image, images_per_sequence, image_path) -> [
    (int, int, int, int)]:
    # Get handles to input and output tensors
    images_in_sequence = images_per_sequence[sequence_per_image[image_path]]
    current_frame_idx = images_in_sequence.index(image_path)
    im_prev_path = image_path if current_frame_idx == 0 else images_in_sequence[current_frame_idx-1]

    current_frame = skimage.io.imread(image_path)
    prev_frame = skimage.io.imread(im_prev_path)
    image_s = cv2.subtract(current_frame, prev_frame)
    image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2GRAY)
    image_s = np.expand_dims(image_s, axis=2)
    four_channels_im = np.concatenate((current_frame, image_s), axis=2)
    return four_channels_im


def generate_build_image_function(labeled_path):
    images_per_sequence = {}
    sequence_per_image = {}
    for folder, images, jsons in find_sequences_with_data(labeled_path):
        images_per_sequence[folder] = images
        for i in images:
            sequence_per_image[i] = folder

    def build_image_f(image_path):
        i = build_image(sequence_per_image, images_per_sequence, image_path)
        return i

    return build_image_f


def main():

    ckpt_path = "/home/seten/TFM/exported_graphs/model_shuttlecock_4_channels_0_junio/frozen_inference_graph.pb"
    labeled_path = "/media/seten/Datos/diego/TFM/dataset_tfm/shuttlecock/secuencias/test"
    images_out_path = '/home/seten/TFM/debug_images/shuttlecock_memoria/faster_4_channels'
    build_image_f = generate_build_image_function(labeled_path)
    min_iou = 0.5
    generic_eval(build_image_f, ckpt_path, labeled_path, images_out_path, min_iou)


if __name__ == '__main__':
    main()
