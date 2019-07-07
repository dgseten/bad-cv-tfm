import os

from PIL import Image

from model.dataset.labelme import LabelMeImageData, LabelMeShapeData
from model.dataset.pascal_voc_io import PascalVocReader

PATH_TO_LABEL_IMAGES_DIR = '/media/seten/Datos/diego/TFM/dataset_tfm/player_poles/test'
OVERWRITE = False


def main():
    xmls_files = [os.path.join(PATH_TO_LABEL_IMAGES_DIR, i) for i in
                  filter(lambda a: a.endswith(".xml"), os.listdir(PATH_TO_LABEL_IMAGES_DIR))]

    print("We are going to convert {} xmls to json".format(len(xmls_files)))

    for i, xml_path in enumerate(xmls_files):
        print("{}/{}".format(i + 1, len(xmls_files)))
        # read labelimg xml
        pascal_xml = PascalVocReader(xml_path)
        im_path = xml_path.replace(".xml", ".jpeg")
        if not os.path.isfile(im_path):
            print("{} does no exists, skipping this file".format(im_path))
            continue

        image = Image.open(im_path)
        width, height = image.size

        labelme_data = LabelMeImageData()
        labelme_data.imagePath = os.path.basename(im_path)
        labelme_data.imageHeight = height
        labelme_data.imageWidth = width

        for shape in pascal_xml.shapes:

            # write objets
            class_name = str(shape[0])
            x_cts = []
            y_cts = []

            for p in shape[1]:
                x_cts.append(p[0])
                y_cts.append(p[1])

            ymin = int(min(y_cts))
            xmin = int(min(x_cts))
            ymax = int(max(y_cts))
            xmax = int(max(x_cts))

            shape_r = LabelMeShapeData()
            shape_r.label = class_name
            shape_r.points = [[xmin, ymin], [xmax, ymax]]
            labelme_data.shapes.append(shape_r)

        output_label_data_json = im_path.replace(".jpeg", ".json")
        if not OVERWRITE and os.path.isfile(output_label_data_json):
            print("Skip save annotations for {}".format(os.path.basename(im_path)))
            return

        labelme_data.serialize_json(output_label_data_json)

        # create labelme data

        # serialize in json

        # generate


if __name__ == "__main__":
    main()
