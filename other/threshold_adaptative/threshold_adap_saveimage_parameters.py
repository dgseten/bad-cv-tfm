import cv2
import cvutils
import os
import itertools


class ThresholdAdaptParameters:
    def __init__(self, origin_path, target_folder_path, maxval, adaptiveMethod, thresholdType, blockSize,c):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.maxval = int(maxval)
        self.adaptiveMethod = int(adaptiveMethod)
        self.thresholdType = thresholdType
        self.blockSize = blockSize
        self.c = c


def test_threslhold_adpt_algorithm_with_parameters(threshold_adapt_params):
    cvutils.create_dir(threshold_adapt_params.target_folder_path)

    frame_output_name = "{}--maxval{}-adaptiveMethod{}-thresholdType{}-blockSize{}-c{}.jpeg".format(
        os.path.basename(threshold_adapt_params.origin_path),
        threshold_adapt_params.maxval, threshold_adapt_params.adaptiveMethod, threshold_adapt_params.thresholdType,
        threshold_adapt_params.blockSize,threshold_adapt_params.c)
    print(frame_output_name)
    full_target_path = os.path.join(threshold_adapt_params.target_folder_path, frame_output_name)

    # test sobel
    img = cv2.imread(threshold_adapt_params.origin_path, 0)
    out = cv2.adaptiveThreshold(img, threshold_adapt_params.maxval, threshold_adapt_params.adaptiveMethod,
                                     threshold_adapt_params.thresholdType, threshold_adapt_params.blockSize,threshold_adapt_params.c)
    # Save resulting image
    cv2.imwrite(full_target_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def main():
    origin_path_list = ["C:\\TFM\imagenes\\finalRioWS-69040-1726.jpeg", "C:\\TFM\\imagenes\\finalRioWS-99000-2475.jpeg"]
    target_folder_path = "C:\\TFM\\ws1\\test_threshold_adaptative\\results"
    maxval_list = [255]
    adaptiveMethod_list = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
    thresholdType_list = [cv2.THRESH_BINARY_INV]
    blockSize_list = [3, 5, 7,11]
    c_list = [2,4,6]

    iterables = [origin_path_list, maxval_list, adaptiveMethod_list, thresholdType_list, blockSize_list,c_list]
    combinations = 0
    for _ in itertools.product(*iterables):
        combinations += 1

    count = 0
    print("Test image  {}/{}".format(count, combinations))
    for t in itertools.product(*iterables):
        test_threslhold_adpt_algorithm_with_parameters(
            ThresholdAdaptParameters(t[0], target_folder_path, t[1], t[2], t[3],t[4],t[5]))
        count += 1
        print("Test image {}/{}".format(count, combinations))


if __name__ == "__main__":
    main()
