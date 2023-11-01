from apt_tele_decode import TelemetryGrabber
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Initialize the grabber with an image and execute  the get channel and visulaize functions
if __name__ == "__main__":
    grabber = TelemetryGrabber("noaa18c.png")
    telemetry, telemetry_raw = grabber.generate_telemetry()
    tel_a = telemetry[0]
    space = (209.80 - 6.25) / 0.8198
    #grabber.visualize_telemetry()
    # gray level calibration of the image
    # lut_out are target values lut_in the ones found in calibration strip
    calib_value_arrangement = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    img = grabber.img
    lut_out = [0, 31, 63, 95, 127, 159, 191, 223, 255]
    lut_in = []
    for i in calib_value_arrangement:
        lut_in.append(tel_a[i])
    lut_8u = np.interp(np.arange(0, 256), lut_in, lut_out).astype(np.uint8)

    img_target = cv2.LUT(img, lut_8u)

    # !!!! importand regenerate telematry and space after image adjustment
    grabber.img = img_target
    telemetry, telemetry_raw = grabber.generate_telemetry()
    #grabber.visualize_telemetry()
    space = grabber.generate_space()[1]
    tel_new = telemetry[1]
    print("")
    # cv2.imshow("Orig", img)
    # cv2.imshow("Adjusted", img_target)
    # cv2.waitKey(0)

    art_tel = np.array([30.39, 57.98, 84.77, 111.31, 137.03, 163.08, 188.95,
                        214.42, 4.09, 95.14, 94.71, 94.30, 94.17, 89.37, 61.17, 111.78])
    tel_new = (art_tel - 6.25) / 0.8198
    #tel_new = np.array([(i - 6.25) / 0.8198 for i in art_tel])
    Cprts = tel_new[9:13]
    # noaa 18 d values for temperature compensation
    ds = np.array([[276.601, 0.05090, 1.657e-06],
                   [276.683, 0.05101, 1.482e-06],
                   [276.565, 0.05117, 1.313e-06],
                   [276.615, 0.05103, 1.484e-06]])
    Tprts = np.zeros(shape=Cprts.shape)
    for i in range(0, 4):
        Tprts[i] = ds[i][0] + ds[i][1] * Cprts[i] * 4 + ds[i][2] * ((Cprts[i] * 4) ** 2)

    Tbb = np.median(Tprts)
    Tw14 = 0.124 * tel_new[13] + 90.113
    # space = grabber.generate_space()

    c1 = 1.1910427e-5
    c2 = 1.4387752

    # temperature-to-radiance coefficients for NOAA-18 Ch 3b Ch 4 Ch 5  in each vc A B
    ttrc = np.array([[2659.7952, 1.698704, 0.996960],
                     [928.1460, 0.436645, 0.998607],
                     [833.2532, 0.253179, 0.999057]])

    Tbbs = ttrc[1][1] + ttrc[1][2] * Tbb
    Nbb = (c1 * (861.89333333333333333333333333333333333333333333333333333333333333 ** 3)) / ((np.e ** ((c2 * ttrc[1][0]) / Tbbs)) - 1)

    # Ns is channel and satalite dependent
    Ns = -5.53  # for ch 4
    Cs = ((209.80 - 6.25) / 0.8198) * 4
    Cbb = tel_new[13] * 4
    Ce = 55 * 4  # image data * 4

    # NOAA-18 Radiance of Space and Coefficients for Nonlinear Radiance Correction Quadratic. Ch4 Ch5, Ns b0 b1 b2
    RsNRC = np.array([[-5.53, 5.82, -0.11069, 0.00052337],
                      [-2.22, 2.67, -0.04360, 0.00017715]])

    # Ns + (Nbb - Ns) * ((Cs - Ce) / (Cs - Cbb))
    Te_l = []
    #for i in range(55, 134):
    Ce = 155.831 * 4
    Nlin = RsNRC[0][0] + (Nbb - RsNRC[0][0]) * ((Cs - Ce) / (Cs - Cbb))
    # only needed for ch 4 5 not 3b
    Ncor = RsNRC[0][1] + RsNRC[0][2] * Nlin + RsNRC[0][3] * (Nlin ** 2)
    Ne = Nlin + Ncor

    # step 4 get blakcbody temp
    Tes = (c2 * ttrc[1][0]) / np.log(1 + ((c1 * (ttrc[1][0] ** 3)) / Ne))
    Te = (Tes - ttrc[1][1]) / ttrc[1][2]
        #Te_l.append(Te)
    # Te is now the black body tempreature at that point

    wdg10 = np.array([[95, 96, 94, 94, 97, 96, 95, 95],
                      [95, 95, 97, 96, 94, 93, 96, 96],
                      [95, 95, 95, 95, 94, 94, 96, 96],
                      [94, 95, 97, 96, 95, 95, 96, 96],
                      [93, 94, 96, 96, 95, 95, 96, 96],
                      [95, 94, 94, 95, 93, 95, 96, 95],
                      [96, 96, 95, 94, 94, 95, 95, 95],
                      [94, 94, 95, 95, 96, 96, 96, 96]])
    wdg11 = np.array([[95, 94, 93, 94, 95, 96, 95, 94],
                      [93, 94, 96, 97, 95, 95, 96, 96],
                      [94, 95, 94, 94, 96, 97, 95, 94],
                      [95, 95, 95, 94, 95, 94, 94, 94],
                      [95, 95, 93, 94, 96, 96, 94, 94],
                      [94, 95, 95, 95, 96, 95, 94, 94],
                      [95, 95, 94, 95, 97, 96, 94, 94],
                      [95, 96, 93, 93, 95, 95, 93, 94]])
    wdg12 = np.array([[96, 95, 93, 94, 95, 95, 94, 94],
                      [95, 94, 93, 94, 97, 96, 94, 93],
                      [93, 94, 97, 96, 94, 94, 94, 95],
                      [94, 94, 95, 95, 94, 94, 94, 94],
                      [94, 95, 94, 94, 94, 94, 94, 95],
                      [95, 94, 93, 94, 95, 95, 94, 95],
                      [93, 94, 95, 95, 95, 94, 95, 95],
                      [93, 93, 95, 95, 94, 94, 94, 95]])
    wdg13 = np.array([[93, 94, 95, 95, 94, 93, 94, 95],
                      [94, 93, 94, 95, 94, 94, 95, 95],
                      [93, 93, 94, 95, 95, 95, 95, 94],
                      [93, 93, 94, 94, 94, 94, 95, 95],
                      [95, 95, 92, 93, 96, 95, 93, 93],
                      [96, 95, 93, 92, 95, 95, 93, 94],
                      [93, 93, 96, 96, 93, 93, 95, 95],
                      [93, 94, 97, 95, 93, 94, 95, 95]])

    wdg14 = np.array([[89, 88, 90, 90, 89, 89, 90, 89],
                      [88, 89, 91, 90, 88, 89, 90, 89],
                      [88, 88, 89, 90, 89, 90, 90, 90],
                      [89, 89, 90, 90, 89, 90, 90, 90],
                      [89, 89, 90, 89, 88, 88, 90, 90],
                      [89, 90, 90, 89, 89, 90, 89, 89],
                      [90, 89, 88, 90, 91, 90, 90, 89],
                      [89, 89, 90, 90, 90, 90, 90, 89]], dtype=np.uint8)

    space = np.array([[211, 211, 206, 206, 211, 210, 208, 209, 211, 211, 206, 206],
                      [211, 212, 207, 207, 211, 211, 208, 208, 211, 211, 206, 206],
                      [211, 211, 206, 207, 211, 210, 208, 208, 210, 211, 207, 207],
                      [211, 209, 205, 207, 211, 211, 208, 209, 212, 211, 206, 206]])

    art_tel = np.array([30.39, 57.98, 84.77, 111.31, 137.03, 163.08, 188.95,
                        214.42, 4.09, 95.14, 94.71, 94.30, 94.17, 89.37, 61.17, 111.78])

    art_tel_cor_me = np.interp(art_tel, lut_in, lut_out).astype(np.float64)
    art_tel_cor_formel = [(i - 6.25) / 0.8198 for i in art_tel]

    Stationc = 0
    AVHRRc = (Stationc - 6.25) / 0.8198

    calib_value_arrangement = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    lut_out = [0, 31, 63, 95, 127, 159, 191, 223, 255]
    lut_in = []
    for i in calib_value_arrangement:
        lut_in.append(art_tel[i])
    art_tel_corr = np.interp(art_tel, lut_in, lut_out).astype(np.float64)

    lut_8u = np.interp(np.arange(0, 256), lut_in, lut_out).astype(np.uint8)

    wdg14_corr = cv2.LUT(wdg14, lut_8u)
    wdg14_corr = np.interp(np.arange(55, 134), lut_in, lut_out).astype(np.float64)

    Ce1 = 623.323
    Ce = 237.863  # 55
    Cs = 993.1690656257624  # space const times 4
    Cbb = 405.56233227616497  # or if not working 427.3127354935946 just all temp fields avg

    Nlin = 115.358
    bch4 = np.array([5.82, -0.11069, 0.00052337])
    Nbb = 88.5174
    # channel 4 in use

    x = np.linspace(20, 100, 100)
    x2 = x * 4
    y = ds[0][0] + ds[0][1] * x2 + ds[0][2] * (x2 ** 2)
    fig = plt.figure()
    # Create the plot
    plt.plot(y, x)

    # Show the plot
    plt.show()
    # !! importand bevor proceding one nedds to calibrate nto only the image but also the telemetry grabed or re do the telemetry

    # temperature-to-radiance coefficients for NOAA-17 Ch 3b Ch 4 Ch 5  in each vc A B
    ttrc = np.array([[2669.3554, 1.702380, 0.997378],
                     [926.2947, 0.271683, 0.998794],
                     [839.8246, 0.309180, 0.999012]])

    # noaa 17 d values for temperature compensation
    ds = np.array([[276.628, 0.05098, 1.371e-06],
                   [276.538, 0.05098, 1.371e-06],
                   [276.761, 0.05097, 1.369e-06],
                   [276.660, 0.05100, 1.348e-06]])
    # black body temperature calibration
