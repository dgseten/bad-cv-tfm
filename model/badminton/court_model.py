import numpy as np
import cv2
import itertools

COURT_INTERSECTION_POINTS = [(0, 0), (609, 0),
                             (3, 3), (46, 3), (49, 3), (303, 3), (306, 3), (560, 3), (563, 3), (606, 3),
                             (3, 76), (46, 76), (49, 76), (303, 76), (306, 76), (560, 76), (563, 76), (606, 76),
                             (3, 79), (46, 79), (49, 79), (303, 79), (306, 79), (560, 79), (563, 79), (606, 79),
                             (3, 468), (46, 468), (49, 468), (303, 468), (306, 468), (560, 468), (563, 468), (606, 468),
                             (3, 471), (46, 471), (49, 471), (303, 471), (306, 471), (560, 471), (563, 471), (606, 471),
                             (3, 868), (46, 868), (49, 868), (303, 868), (306, 868), (560, 868), (563, 868), (606, 868),
                             (3, 871), (46, 871), (49, 871), (303, 871), (306, 871), (560, 871), (563, 871), (606, 871),
                             (3, 1260), (46, 1260), (49, 1260), (303, 1260), (306, 1260), (560, 1260), (563, 1260),
                             (606, 1260),
                             (3, 1260), (46, 1260), (49, 1260), (303, 1260), (306, 1260), (560, 1260), (563, 1260),
                             (606, 1260),
                             (3, 1263), (46, 1263), (49, 1263), (303, 1263), (306, 1263), (560, 1263), (563, 1263),
                             (606, 1263),
                             (3, 1336), (46, 1336), (49, 1336), (303, 1336), (306, 1336), (560, 1336), (563, 1336),
                             (606, 1336),
                             (0, 1339), (609, 1339)
                             ]

COURT_LINES = [
    [(0, 0), (609, 0)],
    [(0, 3), (609, 3)],
    [(0, 76), (609, 76)],
    [(0, 79), (609, 79)],
    [(0, 468), (609, 468)],
    [(0, 471), (609, 471)],
    [(0, 868), (609, 868)],
    [(0, 871), (609, 871)],
    [(0, 1260), (609, 1260)],
    [(0, 1263), (609, 1263)],
    [(0, 1339), (609, 1339)],
    [(0, 1336), (609, 1336)],
    [(0, 0), (0, 1339)],
    [(3, 0), (3, 1339)],
    [(46, 0), (46, 1339)],
    [(49, 0), (49, 1339)],
    [(303, 0), (303, 1339)],
    [(306, 0), (306, 1339)],
    [(560, 0), (560, 1339)],
    [(563, 0), (563, 1339)],
    [(606, 0), (606, 1339)],
    [(609, 0), (609, 1339)]
]


class BadmintonCourt(object):
    def __init__(self):
        pass

    @staticmethod
    def court_image():
        # Create a black image
        img = np.zeros((1340, 610, 1), np.uint8)

        # lineas horizontales de arriba a abajo
        for line in COURT_LINES:
            cv2.line(img, line[0], line[1], 255, 1)

        return img

    @staticmethod
    def court_lines():
        return COURT_LINES

    @staticmethod
    def court_corners():

        horizontal_lines = []
        vertical_lines = []

        # for each line we should estimate de line parameters y = ax + b; a and b are the parameters
        for line in COURT_LINES:

            if line[0][0] == line[1][0]:
                vertical_lines.append(line)
            else:
                horizontal_lines.append(line)

        corners = []
        for hoz_line in horizontal_lines:
            for vert_line in vertical_lines:
                corners.append((vert_line[0][0],hoz_line[0][1]))

        corners.sort(key = lambda point: point[0]+point[1])
        return corners

    @staticmethod
    def court_external_4_corners():
        #return [(0, 0),(0, 1339),(609, 1339), (609, 0)]
        return [(0, 1339), (609, 1339),(0, 0), (609, 0)]

    @staticmethod
    def court_medium_corners():
        aux = []
        for point in BadmintonCourt.court_corners():
            if point[1]> 669:
                aux.append(point)
        return aux

if __name__ == "__main__":
    b = BadmintonCourt()

    cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\court_image.jpg", b.court_image(),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
