import numpy as np
import os
import tensorflow as tf

from model.dataset.labelme import save_labelme_info_from_output_dict
from object_detection.utils import ops as utils_ops
import skimage
import cv2


from object_detection.utils import label_map_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "/home/seten/TFM/exported_graphs/model_shuttlecock_6_channels_3/frozen_inference_graph.pb"
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/seten/TFM/exported_graphs/model_shuttlecock_subtract_0/shuttle_map.pbtxt"
PATH_TO_LABEL_IMAGES_DIR = '/media/seten/Datos/diego/TFM/dataset_tfm/shuttlecock/secuencias/45'
OVERWRITE = False
NUM_CLASSES = 1
# Size, in inches, of the output images.
IMAGE_SIZE = (192, 128)



def main():

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
                        sorted(filter(lambda a: a.endswith(".jpeg"), os.listdir(PATH_TO_LABEL_IMAGES_DIR)))]

    # TODO manage overwrite

    print("We are going to boostrap {} images".format(len(ordered_test_set)))


    list_output_dict = run_inference_for_image_list(ordered_test_set, detection_graph)
    # Visualization of the results of a detection.
    for i, output_dict in enumerate(list_output_dict):
        im_current_path = os.path.join(PATH_TO_LABEL_IMAGES_DIR, ordered_test_set[i])
        save_labelme_info_from_output_dict(output_dict,im_current_path, category_index)

        # generate

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
        """


def run_inference_for_image_list(ordered_test_set, graph):
    output_dict_list = []
    with graph.as_default():
        with tf.Session() as sess:

            for i in range(len(ordered_test_set)):

                im_current_path = os.path.join(PATH_TO_LABEL_IMAGES_DIR, ordered_test_set[i])
                im_prev_path = im_current_path if i == 0 else os.path.join(PATH_TO_LABEL_IMAGES_DIR,
                                                                           ordered_test_set[i - 1])

                current_frame = skimage.io.imread(im_current_path)
                prev_frame = skimage.io.imread(im_prev_path)
                image_s = cv2.subtract(current_frame, prev_frame)
                six_channels_im = np.concatenate((current_frame, image_s), axis=2)

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
                        detection_masks, detection_boxes, image_s.shape[0], image_s.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                print("running {}/{}".format(i + 1, len(ordered_test_set)))
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(six_channels_im, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                output_dict_list.append(output_dict)
    return output_dict_list


if __name__ == "__main__":
    main()
