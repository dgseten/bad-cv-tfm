from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from copy import copy
from pathlib import Path
import pickle
import hashlib

import numpy as np
import os
import tensorflow as tf
import skimage

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import shuttle.utils.dataset_utils as dataset_utils
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_LABELS = "/home/seten/TFM/data/detector_map.pbtxt"
NUM_CLASSES = 2


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def bb_intersection_over_union(box_a: (int, int, int, int), box_b: (int, int, int, int)):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = (x_b - x_a) * (y_b - y_a)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


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


def get_gt_by_class(class_gt: str, json_path: str) -> [(int, int, int, int)]:
    shapes_gt = []
    with open(json_path, "r") as file:
        json_str = file.read()
    data = json.loads(json_str)
    width = int(data['imageWidth'])
    height = int(data['imageHeight'])
    for obj in data['shapes']:
        if obj["label"] != class_gt:
            continue

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
        shapes_gt.append((s_xmin, s_ymin, s_xmax, s_ymax))

    return shapes_gt



"""
                vis_util.visualize_boxes_and_labels_on_image_array(
                    current_frame,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=4)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(current_frame)

                # draw_court_lines_from_detections(image_np, output_dict['detection_boxes'],
                #                                 output_dict['detection_classes'],
                #                                 output_dict['detection_scores'])

                Image.fromarray(current_frame).save(out_debug_image_path)
"""


class TensorflowPredictor(object):

    def __init__(self, pb_model_path, build_image_f, debug_path=None, cache_directory=".tfpredictorcache"):
        self.detection_graph = None
        self.tf_session = None
        self.pb_model_path = pb_model_path
        self.parent_cache_dir = Path(cache_directory)
        m = hashlib.md5()
        m.update(pb_model_path.encode())
        self.cache_dir = self.parent_cache_dir / str(m.hexdigest())
        self.build_image_f = build_image_f
        self.debug_path = debug_path

        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def __enter__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.pb_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.detection_graph = self.detection_graph.as_default()
        self.detection_graph.__enter__()
        self.tf_session = tf.Session()
        self.tf_session.__enter__()

        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        if self.debug_path is not None:
            Path(self.debug_path).mkdir(parents=True,exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tf_session.__exit__(exc_type, exc_val, exc_tb)
        self.detection_graph.__exit__(exc_type, exc_val, exc_tb)

    def get_image_cache_path(self, image_path) -> Path:
        m = hashlib.md5()
        m.update(str(image_path).encode())
        cache_file_name = m.hexdigest()
        cache_path = self.cache_dir / str(cache_file_name)
        return cache_path

    def predict(self, image):
        cache_path = self.get_image_cache_path(image)
        if cache_path.exists():
            with open(str(cache_path), 'rb') as handle:
                return pickle.load(handle)
        else:
            out_predictions = self.get_predictions(image)
            with open(str(cache_path), 'wb') as handle:
                pickle.dump(out_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return out_predictions

    def get_predictions(self, image_path):
        b, s = self.run_inference_for_single_image(image_path, min_score=0.3)
        return b

    def run_inference_for_single_image(self, image_path, min_score):
        # Get handles to input and output tensors
        image = self.build_image_f(image_path)
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = self.tf_session.run(tensor_dict,
                                          feed_dict={image_tensor: np.expand_dims(image, 0)})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        # Filtramos resultados que no sean de la clase player
        index_to_remove = []
        for i,c in enumerate(output_dict['detection_classes']):
            if c!=2:
                index_to_remove.append(i)
        output_dict['num_detections'] -= len(index_to_remove)
        output_dict['detection_classes'] = np.delete(output_dict['detection_classes'],index_to_remove)
        output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], index_to_remove,axis=0)
        output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], index_to_remove)

        # Draw ejemplos visuales
        if self.debug_path is not None:
            current_frame = skimage.io.imread(image_path)
            vis_util.visualize_boxes_and_labels_on_image_array(
                current_frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=4)

            folder_debug_image_path = Path(self.debug_path) / Path(image_path).parent.name
            folder_debug_image_path.mkdir(parents=True,exist_ok=True)
            out_debug_image_path = folder_debug_image_path / Path(image_path).name
            Image.fromarray(current_frame).save(out_debug_image_path)

        # Return boxes..

        return_boxes = []
        return_scores = []

        for i, (b, s) in enumerate(zip(output_dict['detection_boxes'], output_dict['detection_scores'])):
            if s >= min_score:
                b_xmin = b[1] * image.shape[1]
                b_ymin = b[0] * image.shape[0]
                b_xmax = b[3] * image.shape[1]
                b_ymax = b[2] * image.shape[0]
                return_boxes.append((b_xmin, b_ymin, b_xmax, b_ymax))
                return_scores.append(s)

        return return_boxes, return_scores


def generic_eval(get_predictions_f, ckpt_path, labeled_path, images_out_path, min_iou=0.2):
    with TensorflowPredictor(ckpt_path, get_predictions_f,images_out_path) as tf_predictor:
        total_tps = []
        total_fps = []
        total_gts = []
        total_ims = []
        total_seq = 0

        # buscar video
        for sequence_name, images, jsons in find_sequences_with_data(labeled_path):
            print(sequence_name, len(images), len(jsons))

            sequence_tps = []
            sequence_fps = []
            sequence_gts = []
            total_ims.append(len(images))
            total_seq += 1

            # recall plot_per_seq
            recall_plot = []


            for frame_number, (json_path, image_path) in enumerate(zip(jsons, images)):
                gt_boxes = get_gt_by_class("player",json_path)
                prediction_boxes = tf_predictor.predict(image_path)
                prediction_boxes_tp = []
                prediction_boxes_fp = []

                pending_gt_boxes = copy(gt_boxes)
                for p_box in prediction_boxes:
                    ious = [bb_intersection_over_union(p_box, gt_box) for gt_box in pending_gt_boxes]
                    best_iou = max(ious) if len(ious) > 0 else 0
                    if best_iou >= min_iou:
                        pending_gt_boxes.pop(ious.index(best_iou))
                        prediction_boxes_tp.append(p_box)
                    else:
                        prediction_boxes_fp.append(p_box)

                # Count_results
                im_tp = len(gt_boxes) - len(pending_gt_boxes)
                im_fp = len(prediction_boxes) - im_tp
                print("-" * 80)
                print(f"Results image {frame_number} (IOU={min_iou}) {image_path}, tp: {im_tp}, fp: {im_fp}, "
                      f"predicted/relevant elements: {len(prediction_boxes), len(gt_boxes)}")

                precc = (im_tp / len(prediction_boxes)) if len(prediction_boxes) > 0 else 1
                recall = (im_tp / len(gt_boxes)) if len(gt_boxes) > 0 else 1

                print(f"Precision: {'%.2f' % (precc)}")
                print(f"Recall: {'%.2f' % (recall)}")
                sequence_tps.append(im_tp)
                sequence_fps.append(im_fp)
                sequence_gts.append(len(gt_boxes))
                recall_plot.append(im_tp)


            print("=" * 80)
            print("=" * 80)
            print(f"Results for sequence {sequence_name}:")
            print(f"- images: {len(images)}")
            print(f"- tp: {sum(sequence_tps)}")
            print(f"- fp: {sum(sequence_fps)}")
            print(f"- gt: {sum(sequence_gts)}")
            print(f"- precision: {'%.2f' % (sum(sequence_tps) / (sum(sequence_tps) + sum(sequence_fps)))}")
            print(f"- recall: {'%.2f' % (sum(sequence_tps) / sum(sequence_gts))}")
            print("=" * 80)

            total_fps.append(sum(sequence_fps))
            total_tps.append(sum(sequence_tps))
            total_gts.append(sum(sequence_gts))

        print("*" * 80)
        print("*" * 80)
        print(f"Global results (IOU={min_iou}): {labeled_path}")
        print(f"- images: {sum(total_ims)}")
        print(f"- sequences: {total_seq}")
        print(f"- tp: {sum(total_tps)}")
        print(f"- fp: {sum(total_fps)}")
        print(f"- gt: {sum(total_gts)}")
        print(f"- precision: {'%.2f' % (sum(total_tps) / (sum(total_tps) + sum(total_fps)))}")
        print(f"- recall: {'%.2f' % (sum(total_tps) / sum(total_gts))}")
        print("*" * 80)
