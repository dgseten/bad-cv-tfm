from model.badminton.court_model import BadmintonCourt
import numpy as np
import cv2


if __name__ == "__main__":


    img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')

    c = BadmintonCourt()
    ground_truth_corners = c.court_external_4_corners()
    fixed_points_example = [(620,498),(369,1019),(1548,1019),(1288,498)]

    src_pts = np.array(ground_truth_corners, np.float32)
    dst_pts = np.array(fixed_points_example, np.float32)

    M, mask = cv2.findHomography(src_pts, dst_pts)
    matchesMask = mask.ravel().tolist()

    h = 1340
    w = 610

    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


    cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\final_court_example_finder.jpg", img2,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
