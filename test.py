import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
x = np.linspace(0, np.pi, 800)
intensity_values = np.sin(x)

# https://stackoverflow.com/a/35881382/190597
fig, (ax0, ax1) = plt.subplots(
    nrows=2, gridspec_kw={'height_ratios':[7, 1],}, sharex=True)
ax0.plot(x, intensity_values)
ax1.imshow(np.atleast_2d(intensity_values), cmap=plt.get_cmap('gray'),
              extent=(0, np.pi, 0, 1))
plt.show()