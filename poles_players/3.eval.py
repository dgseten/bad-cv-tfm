import skimage
from poles_players.gen_eval_preccision_recall_model import generic_eval


def build_image(image_path):
    return skimage.io.imread(image_path)


def main():
    build_image_f = build_image
    ckpt_path = "/home/seten/TFM/exported_graphs/model_player_poles_0_junio/frozen_inference_graph.pb"
    labeled_path = "/media/seten/Datos/diego/TFM/dataset_tfm/player_poles"
    images_out_path = '/home/seten/TFM/debug_images/player_memoria/faster_simple'
    min_iou = 0.5
    generic_eval(build_image_f, ckpt_path, labeled_path, images_out_path, min_iou)


if __name__ == '__main__':
    main()
