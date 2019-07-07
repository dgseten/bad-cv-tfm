import cv2
import numpy as np
import itertools

img = cv2.imread('C:\\TFM\\ws1\\test_final_alg\\court_pattern_2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\threshold.jpg", thresh3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#edges = cv2.Canny(thresh3, 100, 200, apertureSize=3)
#cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\canny.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


"""
line ecuation: y = m*x +n
polar:

    punto(3,6): indica que estamos a una distancia de 3 desde el origen de coordenadas con un angulo de 6

    p = x * cos(o) + y* sin(o)
    p =

"""

lines = cv2.HoughLines(thresh3, 1, np.pi / 180, 200)
line_equation_params = []
shape = img.shape
img_rows = shape[0]
img_cols = shape[1]

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        line_equation_params.append([[a, b], rho])

# calculate intersection for all lines
# se puede hacer también con dos bucles for.
combinations = itertools.combinations(line_equation_params, 2)
intersections = []
for combination in combinations:

    a = np.array([combination[0][0], combination[1][0]])
    b = np.array([combination[0][1], combination[1][1]])
    x = None
    try:
        x = np.linalg.solve(a, b)
        x_coord = x[0]
        y_coord = x[1]

        if x_coord >= 0 and y_coord >= 0 and x_coord <= img_cols and y_coord <= img_rows:
            intersections.append((int(x_coord),int(y_coord)))
    except:
        # parallel lines
        pass

# TODO temporal draw intersections
for intersection in intersections:
    cv2.circle(img,intersection,3, (0,0,255), -1)


cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\hough_canny_100_200_1_180_250.jpg", img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
