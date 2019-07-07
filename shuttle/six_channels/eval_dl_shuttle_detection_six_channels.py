import numpy as np
import os
import tensorflow as tf
from PIL import Image
import PIL.ImageDraw as ImageDraw
# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops
from model.badminton.court_model import BadmintonCourt
import skimage
from matplotlib import pyplot as plt
import cv2


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "/home/seten/TFM/exported_graphs/model_shuttlecock_6_channels_3/frozen_inference_graph.pb"
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/seten/TFM/exported_graphs/model_shuttlecock_subtract_0/shuttle_map.pbtxt"
PATH_TO_TEST_IMAGES_DIR = '/media/seten/Datos/diego/TFM/dataset_tfm/shuttlecock/secuencias/40'
OUT_PATH_EVAL_IMAGES = "/home/seten/TFM/debug_images/model_shuttlecock_6_channels_0/concat_1_40"
NUM_CLASSES = 1
# Size, in inches, of the output images.
IMAGE_SIZE = (192, 128)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def main():
    print("Creating eval directory")
    os.makedirs(OUT_PATH_EVAL_IMAGES, exist_ok=True)

    # load frozen graph in memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    ordered_test_set = [i for i in
                        sorted(filter(lambda a: a.endswith(".jpeg"), os.listdir(PATH_TO_TEST_IMAGES_DIR)))]
    print("We are going to run the inference for {} images".format(len(ordered_test_set)))

    with detection_graph.as_default():
        with tf.Session() as tf_session:

            for i in range(len(ordered_test_set)):
                print("running {}: {}".format(i, ordered_test_set[i]))
                im_current_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, ordered_test_set[i])
                im_prev_path = im_current_path if i == 0 else os.path.join(PATH_TO_TEST_IMAGES_DIR, ordered_test_set[i - 1])

                current_frame = skimage.io.imread(im_current_path)
                prev_frame = skimage.io.imread(im_prev_path)
                image_s = cv2.subtract(current_frame, prev_frame)
                #image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2GRAY)
                #image_s = np.expand_dims(image_s, axis=2)
                four_channels_im = np.concatenate((current_frame,image_s),axis=2)

                # Image.fromarray(image_s).save("tmp.jpeg")

                out_debug_image_path = os.path.join(OUT_PATH_EVAL_IMAGES, os.path.basename(im_current_path))
                # if os.path.isfile(out_debug_image_path):
                #    continue

                # image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_s, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(four_channels_im, tf_session)
                # Visualization of the results of a detection.

                # draw only poles

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


def run_inference_for_single_image(image, tf_session):

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
        print("we have masks! :)")
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
    output_dict = tf_session.run(tensor_dict,
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


def order_corner_points_for_homography(corners_points):
    """

    :param corners_points:
    :return:
    """
    # order by y
    sorted_y_corners = sorted(corners_points, key=lambda p: p[1])
    top_left = sorted(sorted_y_corners[:2], key=lambda p: p[0])[0]
    top_right = sorted(sorted_y_corners[:2], key=lambda p: p[0])[1]
    bottom_left = sorted(sorted_y_corners[2:4], key=lambda p: p[0])[0]
    bottom_right = sorted(sorted_y_corners[2:4], key=lambda p: p[0])[1]
    return [bottom_left, bottom_right, top_left, top_right]


def draw_court_lines_from_detections(image,
                                     boxes,
                                     classes,
                                     scores,
                                     min_score_thresh=.5):
    # first we have to find the corners. in (x,y) format
    image_height, image_witdh, channels = image.shape
    corners_points = []
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            ymin = ymin * image_height
            ymax = ymax * image_height
            xmax = xmax * image_witdh
            xmin = xmin * image_witdh
            corners_points.append((int((xmax + xmin) / 2), int((ymax + ymin) / 2)))

    # now we are going to paint de points.
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    for p in corners_points:
        r = 2
        draw.ellipse((720 - r, 100 - r, 720 + r, 100 + r), fill=(255, 0, 0))
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(255, 0, 0))
        draw.point(p, fill="yellow")

    def print_court_polylines(best_homography_matrix):

        for line in BadmintonCourt.court_lines():
            pts = np.float32([line]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, best_homography_matrix)
            draw.line([(dst[0][0][0], dst[0][0][1]), (dst[1][0][0], dst[1][0][1])], fill="red")

    if len(corners_points) == 4:
        # find homography matrix
        court = BadmintonCourt()
        ground_truth_corners = court.court_external_4_corners()
        src_pts = np.array(ground_truth_corners, np.float32)
        # puede que tengan que tener el mismo orden
        cp = corners_points
        # TODO test si puedo cambiar la lista directamente.

        homography_candidates = order_corner_points_for_homography(corners_points)
        dst_pts = np.array(homography_candidates, np.float32)
        M, mask = cv2.findHomography(src_pts, dst_pts)

        print_court_polylines(M)
        homography_ok_points = 0

    np.copyto(image, np.array(image_pil))


if __name__ == "__main__":
    main()
