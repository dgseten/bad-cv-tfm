import json


def get_image_size_from_annotation_json(json_path):
    with open(json_path, "r") as file:
        json_str = file.read()
    data = json.loads(json_str)
    width = int(data['imageWidth'])
    height = int(data['imageHeight'])

    return width, height


def get_shapes_from_annotation_json(json_path):
    with open(json_path, "r") as file:
        json_str = file.read()
    data = json.loads(json_str)

    width = int(data['imageWidth'])
    height = int(data['imageHeight'])
    shapes = []
    classes = []

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

        shapes.append(((s_xmin, s_ymin), (s_xmax, s_ymax)))
        classes.append(obj["label"])
    return classes, shapes
