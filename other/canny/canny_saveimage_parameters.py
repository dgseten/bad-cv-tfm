import cv2
import cvutils
import csv
import os


class CannyParameters:
    def __init__(self, origin_path, target_folder_path, threshold1, threshold2, apertureSize, l2gradient):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.threshold1 = int(threshold1)
        self.threshold2 = int(threshold2)
        self.aperture_size = int(apertureSize)
        self.l2gradient = l2gradient=='1'


def read_csv(csv_path):
    video_split_list = []
    spam_reader = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|')
    for row in spam_reader:
        video_slit = CannyParameters(row[0], row[1], row[2], row[3], row[4], row[5])
        video_split_list.append(video_slit)
    return video_split_list


def test_canny_algorithm_with_parameters(canny_params, debug=False):
    cvutils.create_dir(canny_params.target_folder_path)

    frame_output_name = "{}-threshold1{}-threshold2{}-apertureSize{}-l2gradient{}.jpeg".format(
        os.path.basename(canny_params.origin_path),
        canny_params.threshold1, canny_params.threshold2, canny_params.aperture_size, canny_params.l2gradient)
    print(frame_output_name)
    full_target_path = os.path.join(canny_params.target_folder_path, frame_output_name)

    # test sobel
    img = cv2.imread(canny_params.origin_path, 0)
    out = cv2.Canny(img, threshold1=canny_params.threshold1, threshold2=canny_params.threshold2,
                    apertureSize=canny_params.aperture_size, L2gradient=canny_params.l2gradient)
    # Save resulting image
    cv2.imwrite(full_target_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main():
    list_split = read_csv("C:\\TFM\\ws1\\test_canny\\test.csv")
    count = 0
    print("Test image  {}/{}".format(count, len(list_split)))
    for split_video_conf in list_split:
        test_canny_algorithm_with_parameters(split_video_conf)
        count += 1
        print("Test image {}/{}".format(count, len(list_split)))


if __name__ == "__main__":
    main()
