import cv2
import cvutils
import os
import itertools




class ThresholdParameters:
    def __init__(self, origin_path, target_folder_path, threshold, maxval,type):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.threshold = int(threshold)
        self.maxval = int(maxval)
        self.type = int(type)


def test_threslhold_algorithm_with_parameters(threshold_params):
    cvutils.create_dir(threshold_params.target_folder_path)

    frame_output_name = "{}-threshold{}-maxval{}-type{}.jpeg".format(
        os.path.basename(threshold_params.origin_path),
        threshold_params.threshold, threshold_params.maxval, threshold_params.type)
    print(frame_output_name)
    full_target_path = os.path.join(threshold_params.target_folder_path, frame_output_name)

    # test sobel
    img = cv2.imread(threshold_params.origin_path, 0)
    ret, out = cv2.threshold(img,threshold_params.threshold,threshold_params.maxval,threshold_params.type)
    # Save resulting image
    cv2.imwrite(full_target_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main():

    origin_path_list = ["C:\\TFM\imagenes\\finalRioWS-69040-1726.jpeg","C:\\TFM\\imagenes\\finalRioWS-99000-2475.jpeg"]
    target_folder_path = "C:\\TFM\\ws1\\test_threshold\\results"
    threshold_list = [240,200,160,140,120]
    max_val_list = [255]
    type_list = [cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV]
    iterables = [origin_path_list,threshold_list, max_val_list,type_list]
    combinations = 0
    for _ in itertools.product(*iterables):
        combinations +=1

    count = 0
    print("Test image  {}/{}".format(count, combinations))
    for t in itertools.product(*iterables):
        test_threslhold_algorithm_with_parameters(ThresholdParameters(t[0],target_folder_path,t[1],t[2],t[3]))
        count += 1
        print("Test image {}/{}".format(count, combinations))


if __name__ == "__main__":
    main()
