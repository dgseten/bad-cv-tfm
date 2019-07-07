import cv2
import cvutils
import csv
import os


class LaplacianParameters:
    def __init__(self, origin_path,target_folder_path,  ksize):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.ksize = int(ksize)


def read_csv(csv_path):
    video_split_list = []
    spam_reader = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|')
    for row in spam_reader:
        video_slit = LaplacianParameters(row[0], row[1], row[2])
        video_split_list.append(video_slit)
    return video_split_list


def test_sobel_algorithm_with_parameters(laplace_param, debug=False):
    cvutils.create_dir(laplace_param.target_folder_path)

    frame_output_name = "{}-ksize{}.jpeg".format(os.path.basename(laplace_param.origin_path),laplace_param.ksize)
    full_target_path = os.path.join(laplace_param.target_folder_path, frame_output_name)

    # test sobel
    img = cv2.imread(laplace_param.origin_path, 0)
    out =cv2.Laplacian(img, cv2.CV_64F,ksize=laplace_param.ksize)
    # Save resulting image
    cv2.imwrite(full_target_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main():
    list_split = read_csv("C:\\TFM\\ws1\\test_laplaciana\\test.csv")
    count = 0
    print("Test image  {}/{}".format(count, len(list_split)))
    for split_video_conf in list_split:
        test_sobel_algorithm_with_parameters(split_video_conf)
        count += 1
        print("Test image {}/{}".format(count, len(list_split)))


if __name__ == "__main__":
    main()
