import matplotlib.pyplot as plt
import cv2
import skimage as ski
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.util import random_noise


original = ski.io.imread("raw0.png")

sigma = 0.155
noisy = original

fig, ax = plt.subplots(nrows=2, ncols=2,
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')


ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, channel_axis=-1))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
print('Done2')
#ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
#                channel_axis=None))
#ax[0, 2].axis('off')
#ax[0, 2].set_title('Bilateral')
print('Done3')
ax[1, 0].imshow(denoise_wavelet(noisy, channel_axis=-1, rescale_sigma=True))
ax[1, 0].axis('off')
ax[1, 0].set_title('Wavelet denoising')
print('Done4')
ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.2, channel_axis=-1))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
print('Done5')
#ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15,
#                channel_axis=None))
#ax[1, 2].axis('off')
#ax[1, 2].set_title('(more) Bilateral')
print('Done6')

ax[0, 0].imshow(original)
ax[0, 0].axis('off')
ax[0, 0].set_title('Original')

print('Done Final')
fig.tight_layout()

plt.show()
