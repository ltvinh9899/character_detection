import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('digits.png',0) # đọc ảnh


# cv2.imshow('sdasf', img)

cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)] # cắt ảnh thành từng phần tử
# print(cells[5][0])

# cv2.imwrite('so_duoi_2.png', cells[5][0])

cells_array = np.array(cells) # chuyển ảnh về dạng mảng

# print(cells_array[0][0])
# chuyển mảng về mảng 1 chiều để tạo dữ liệu train và test
train = cells_array[:,:].reshape(-1,400).astype(np.float32)
# test = cells_array[:, 25:50].reshape(-1,400).astype(np.float32)

# print(train)

labels = np.arange(10) # tạo nhãn

# print(k)

train_labels = np.repeat(labels, 250)[:, np.newaxis] # dán nhãn

knn = cv2.ml.KNearest_create()
knn.train(train,0,train_labels)


img_test = cv2.imread('anh_test_2.png', 0) # đọc ảnh test
print(img_test)
img_test_array = np.array(img_test)
img_test_array_reshape =img_test_array.reshape(-1,400).astype(np.float32)


temp, result, neighbour, distance = knn.findNearest(img_test_array_reshape, 7)

print(temp)
print(result)
print(neighbour)
print(distance)
# print(train_labels[300])

# print(train_labels)
