import cv2
import numpy as np
import itertools



img = cv2.imread('C:\\TFM\\imagenes\\finalRioWS-35880-897.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\threshold.jpg", thresh3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
edges = cv2.Canny(thresh3, 100, 200, apertureSize=3)
# cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\canny.jpg", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


"""
line ecuation: y = m*x +n
polar:

    punto(3,6): indica que estamos a una distancia de 3 desde el origen de coordenadas con un angulo de 6

    p = x * cos(o) + y* sin(o)
    p =

"""

lines = cv2.HoughLines(thresh3, 1, np.pi / 180, 400)
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
            intersections.append([int(x_coord),int(y_coord)])
    except:
        # parallel lines
        pass

#### HOMOGRAPHY

if len(intersections)>25:

    dst_pts = np.array(find_points(),np.float32)
    src_pts = np.array(intersections[:len(dst_pts)], np.float32)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    shape = img.shape
    h = shape[0]
    w = shape[1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)



cv2.imwrite("C:\\TFM\\ws1\\test_final_alg\\final.jpg", img2,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
