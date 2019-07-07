import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = "/home/seten/TFM/exported_graphs/model_player_pole_1/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/seten/TFM/exported_graphs/model_player_pole_1/detector_map.pbtxt"

PATH_TO_TEST_IMAGES_DIR = '/media/seten/Datos/diego/TFM/dataset_tfm/court_poles'

NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# detection

TEST_IMAGE_PATHS = []
for im_file in os.listdir(PATH_TO_TEST_IMAGES_DIR):
    # print(im_file)
    if im_file.endswith(".jpeg") and not os.path.isfile(
            os.path.join(PATH_TO_TEST_IMAGES_DIR, im_file.replace(".jpeg", ".xml"))):
        TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, im_file))
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 8) ]
print(len(TEST_IMAGE_PATHS))


# Size, in inches, of the output images.

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
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
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def generate_pascal_xml(boxes, classes, scores, category_index, input_image_path, output_xml, min_score_thresh=.5):
    from pascal_voc_writer import Writer
    objects_to_include = []
    # filter by score
    image = Image.open(image_path)
    width, height = image.size
    writer = Writer(input_image_path, width, height)
    for object_index in range(len(scores)):
        # filter bad detections-
        if scores[object_index] < min_score_thresh:
            continue
        # write objets
        class_name = str(category_index[classes[object_index]]["name"])
        box = boxes[object_index]
        ymin = int(min(box[0], box[2]) * height)
        xmin = int(min(box[1], box[3]) * width)
        ymax = int(max(box[0], box[2]) * height)
        xmax = int(max(box[1], box[3]) * width)

        writer.addObject(class_name, xmin, ymin, xmax, ymax)

    print("We are going to save pascal xml in {}".format(output_xml))
    writer.save(output_xml)


subset_test = TEST_IMAGE_PATHS[:]

print("We are going to run the inference for {} images".format(len(subset_test)))
for image_path in subset_test:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=3)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    dirname = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    out_path = os.path.join(dirname, "ex_out", image_name)

    print("Saving result image in {}".format(out_path))

    Image.fromarray(image_np).save(out_path)
    output_xml_path = os.path.join(dirname, "ex_out", image_name.split(".")[0]+".xml")

    generate_pascal_xml(output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        image_path,
                        output_xml_path)
