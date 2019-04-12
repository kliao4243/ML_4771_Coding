import scipy.io
from matplotlib import pyplot as PLT
import numpy as np
#from PIL import Image

mat = scipy.io.loadmat('hw2_data.mat')
X1 = mat['X1']
Y1 = mat['Y1']
X2 = mat['X2']
Y2 = mat['Y2']

X1 = X1.astype(np.int)
Y1 = Y1.astype(np.int)
X2 = X2.astype(np.int)
Y2 = Y2.astype(np.int)


def RGB_display(coordinate, RGB_value):
    x_max = 0
    y_max = 0
    for i in range(coordinate.shape[0]):
        if coordinate[i][0] > x_max:
            x_max = coordinate[i][0]
        if coordinate[i][1] > y_max:
            y_max = coordinate[i][1]
    temp_pic = np.zeros((y_max+1, x_max+1, 3), dtype=np.int)
    for i in range(coordinate.shape[0]):
        temp_pic[coordinate[i][1]][coordinate[i][0]] = RGB_value[i]
    temp_pic = temp_pic.reshape((133, 140, 3))
    return temp_pic

test_pic = RGB_display(X2, Y2)
PLT.imshow(test_pic)
PLT.show()
