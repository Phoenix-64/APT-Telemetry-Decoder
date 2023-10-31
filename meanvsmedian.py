import numpy as np
import matplotlib.pyplot as plt

import cv2

img = cv2.imread("raw.png", 0)

tel = img[0:img.shape[0], 2040:2080][:-1]
tel_mean = np.mean(tel, axis=1)
tel_median = np.median(tel, axis=1)

fig, (ax, ax1) = plt.subplots(2)
ax.imshow(np.atleast_2d(tel_mean), cmap=plt.get_cmap('gray'), extent=(0, len(tel), 0, len(tel) / 4))
ax1.imshow(np.atleast_2d(tel_median), cmap=plt.get_cmap('gray'), extent=(0, len(tel), 0, len(tel) / 4))

plt.show()