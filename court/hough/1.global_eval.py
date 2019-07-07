from joblib import Parallel, delayed
from court.hough.court_finder import find_court_lines
import os


def eval_img(img_path):
    try:
        print("Launching_eval for img: {}".format(img_path))
        find_court_lines(img_path)
        return 0
    except Exception as ex:
        print("{}: {}".format(os.path.basename(img_path),ex))
        return -1


def eval(folder_path: str):
    print("Evaluating imgs in folder: {}".format(folder_path))
    imgs = map(lambda im: os.path.join(folder_path,im),filter(lambda im: im.split('.')[-1] == "jpeg", os.listdir(folder_path)))
    imgs = [i for i in imgs]
    print("We will eval {} imgs".format(len(imgs)))
    results = Parallel(n_jobs=6)(delayed(eval_img)(im) for im in imgs)
    print(results)

if __name__ == '__main__':
    eval("/media/seten/Datos/diego/TFM/dataset_tfm/court/test")
