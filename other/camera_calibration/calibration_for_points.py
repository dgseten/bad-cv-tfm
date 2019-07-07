import numpy as np

model_coordenates = [(0, 0, 0), (0, 1339, 0), (609, 1339, 0), (609, 0, 0), (0, 669, 155), (609, 669, 155)]
image_coordenates = [(620, 498), (369, 1019), (1548, 1019), (1288, 498), (531, 472), (1381, 472)]

# create equation as numpy style

a = []
b = []

for i in range(len(model_coordenates)):
    X = model_coordenates[i][0]
    Y = model_coordenates[i][1]
    Z = model_coordenates[i][2]
    u = image_coordenates[i][0]
    v = image_coordenates[i][1]
    a.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z])
    a.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z])
    b.append(u)
    b.append(v)

a = np.array(a)
b = np.array(b)

# matrix rank
print(np.linalg.matrix_rank(a))



#x = np.linalg.solve(a, b)
x = np.linalg.lstsq(a, b)
print(x[0])
sol = x[0]
M = [[sol[0],sol[1],sol[2],sol[3]],[sol[4],sol[5],sol[6],sol[7]],[sol[8],sol[9],sol[10],1]]

#print(np.allclose(np.dot(a, x), b))


# check solutiom foo
print("vertice superior izquierdo de la pista está en: {}".format(np.dot(M,np.array([0,0,0,1]))))
net_point = np.dot(M,np.array([0, 669, 155,1]))
net_point = [net_point[0]/net_point[2],net_point[1]/net_point[2]]
print("vertice superior izquierdo de la red está en: {}".format(net_point))

court_center = np.dot(M,np.array([305, 669, 0,1]))
court_center = [court_center[0]/court_center[2],court_center[1]/court_center[2]]
print("centro de la pista: {}".format(court_center))

court_center = np.dot(M,np.array([305, 669, 155,1]))
court_center = [court_center[0]/court_center[2],court_center[1]/court_center[2]]
print("centro de la red: {}".format(court_center))


court_center = np.dot(M,np.array([305, 669, 200,1]))
court_center = [court_center[0]/court_center[2],court_center[1]/court_center[2]]
print("centro a 2 metros de altura: {}".format(court_center))


# Ahora calculamos la pseudoinversa:

Mi = np.linalg.pinv(M)
print (Mi)
net_point = np.dot(Mi,np.array([531, 472, 1])) # número de columnas de la primera debe coincidir con el número de filas de la segunda
#net_point = [net_point[0]/net_point[3],net_point[1]/net_point[3],net_point[2]/net_point[3]]
print("vertice superior izquierdo de la red en el mundo real está en: {}".format(net_point))


