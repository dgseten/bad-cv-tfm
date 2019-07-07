import cv2
import cvutils
import os
import itertools




class CannyParameters:
    def __init__(self, origin_path, target_folder_path, threshold1, threshold2, apertureSize, l2gradient):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.threshold1 = int(threshold1)
        self.threshold2 = int(threshold2)
        self.aperture_size = int(apertureSize)
        self.l2gradient = l2gradient=='1'


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

    origin_path_list = ["C:\\TFM\imagenes\\finalRioWS-69040-1726.jpeg"]
    target_folder_path = "C:\\TFM\\ws1\\test_canny\\results"
    threshold1_list = [0,100,200,300,400,500,600,700,800,900]
    threshold2_list = [0,100,200,300,400,500,600,700,800,900]
    aperture_size_list = [3]
    l2gradient_list = ['1','0']
    iterables = [origin_path_list,threshold1_list, threshold2_list,aperture_size_list,l2gradient_list]
    combinations = 0
    for _ in itertools.product(*iterables):
        combinations +=1

    count = 0
    print("Test image  {}/{}".format(count, combinations))
    for t in itertools.product(*iterables):
        test_canny_algorithm_with_parameters(CannyParameters(t[0],target_folder_path,t[1],t[2],t[3],t[4]))
        count += 1
        print("Test image {}/{}".format(count, combinations))


if __name__ == "__main__":
    main()
