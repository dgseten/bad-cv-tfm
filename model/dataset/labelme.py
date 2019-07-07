import os
import json
import numpy as np

from PIL import Image


class LabelMeImageData(object):

    def __init__(self):
        self.version = "3.6.16"
        self.flags = {}
        self.shapes = []
        self.lineColor = [0, 255, 0, 128]
        self.fillColor = [255, 0, 0, 128]
        self.imagePath = None
        self.imageData = None
        self.imageHeight = None
        self.imageWidth = None

    def serialize_json(self, path):
        with open(path, "w") as file:
            json.dump(self.__dict__, file, default=serialize, indent=4, sort_keys=True)



class LabelMeShapeData(object):
    def __init__(self):
        self.label = None
        self.line_color = None
        self.fill_color = None
        self.points = []
        self.shape_type = "rectangle"


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    return obj.__dict__


def save_labelme_info_from_output_dict(output_dict, im_current_path, category_index, min_score_thresh=.5, overwrite=False):
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']

    # filter by score
    image = Image.open(im_current_path)
    width, height = image.size

    labelme_data = LabelMeImageData()
    labelme_data.imagePath = os.path.basename(im_current_path)
    labelme_data.imageHeight = height
    labelme_data.imageWidth = width

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
        shape_r = LabelMeShapeData()
        shape_r.label = class_name
        shape_r.points = [[xmin, ymin], [xmax, ymax]]
        labelme_data.shapes.append(shape_r)

    output_label_data_json = im_current_path.replace(".jpeg", ".json")
    if not overwrite and os.path.isfile(output_label_data_json):
        print("Skip save annotations for {}".format(os.path.basename(im_current_path)))
        return

    labelme_data.serialize_json(output_label_data_json)


