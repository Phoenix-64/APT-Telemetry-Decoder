from apt_tele_decode import TelemetryGrabber
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

a_pos = (88, 992)
b_pos = (1128, 2033)
vertical = (115, 1060)
# Initialize the grabber with an image and execute  the get channel and visulaize functions
if __name__ == "__main__":
    grabber = TelemetryGrabber("noaa18c.png", 0)
    #grabber.visualize_telemetry()
    img_org = grabber.img
    Te = grabber.temp_calib()
    img = grabber.img

    img_a = img[:, a_pos[0]:a_pos[1]]
    img_b = img[:, b_pos[0]:b_pos[1]]

    img_a_org = img_org[:, a_pos[0]:a_pos[1]]
    img_b_org = img_org[:, b_pos[0]:b_pos[1]]

    for idx, x in enumerate(Te[1]):
        if np.isnan(x):
            Te[1, idx] = Te[1, idx - 1]
        else:
            Te[1, idx] = x


    lut_out = Te[1]
    lut_in = np.arange(0, 256)
    lut_8u = np.interp(np.arange(0, 256), lut_in, lut_out).astype(np.uint8)

    temp_img_b = cv2.LUT(img_b, lut_8u)
    temp_img_b_dir = Te[1][img_b] - 273.15

    temp_img_b_crop = temp_img_b_dir[vertical[0]:vertical[1]]
    temp_filtered = cv2.bilateralFilter(temp_img_b_crop.astype(np.float32),5,75,75)
    temp_median = cv2.medianBlur(temp_img_b_crop.astype(np.float32), 5)
    temp_nimeans = cv2.fastNlMeansDenoising(temp_img_b_crop.astype(np.uint8), None, 1, 7, 11)


    cmap1 = plt.get_cmap("gray")
    cmap2 = plt.get_cmap("jet")
    fig, ax = plt.subplots(2,2)
    #ax[0][0].imshow(img_a_org, cmap=cmap1)
    #ax[0][1].imshow(img_b_org, cmap=cmap1)
    #ax[1][0].imshow(img_a, cmap=cmap1)
    #ax[1][1].imshow(img_b, cmap=cmap1)
    mapb = ax[0, 0].imshow(temp_img_b_crop, cmap=cmap2)
    mapc = ax[0, 1].imshow(temp_filtered, cmap=cmap2)
    mapa = ax[1, 0].imshow(temp_median, cmap=cmap2)
    ax[1, 1].imshow(temp_nimeans, cmap=cmap2)

    #fig.colorbar(mapb)
    #fig.colorbar(mapc)

    plt.show()
    print(Te)
    #grabber.visualize_telemetry()
