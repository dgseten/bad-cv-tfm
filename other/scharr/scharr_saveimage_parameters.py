import cv2
import cvutils
import csv
import os


class ScharrParameters:
    def __init__(self, origin_path, target_folder_path, ddepth, dx, dy):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.ddepth = int(ddepth)
        self.dx = int(dx)
        self.dy = int(dy)


def read_csv(csv_path):
    video_split_list = []
    spam_reader = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|')
    for row in spam_reader:
        video_slit = ScharrParameters(row[0], row[1], row[2], row[3], row[4])
        if len(row) > 5:
            video_slit.codec = row[5]
        video_split_list.append(video_slit)
    return video_split_list


def test_sobel_algorithm_with_parameters(scharr_params, debug=False):
    cvutils.create_dir(scharr_params.target_folder_path)

    frame_output_name = "{}-dx{}-dy{}.jpeg".format(os.path.basename(scharr_params.origin_path),
                                                   scharr_params.dx,scharr_params.dy)
    full_target_path = os.path.join(scharr_params.target_folder_path, frame_output_name)

    # test sobel
    img = cv2.imread(scharr_params.origin_path, 0)
    sobel = cv2.Scharr(img, 6, scharr_params.dx, scharr_params.dy)
    # Save resulting image
    cv2.imwrite(full_target_path, sobel, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main():
    list_split = read_csv("C:\\TFM\\ws1\\test_scharr\\test.csv")
    count = 0
    print("Test image  {}/{}".format(count, len(list_split)))
    for split_video_conf in list_split:
        test_sobel_algorithm_with_parameters(split_video_conf)
        count += 1
        print("Test image {}/{}".format(count, len(list_split)))


if __name__ == "__main__":
    main()
