from apt_tele_decode import TelemetryGrabber
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


# compute the diffrence when selecting wrong satalite
if __name__ == "__main__":
    List = []
    for i in range(3):
        grabber = TelemetryGrabber("noaa18_2c.png", i)
        List.append(grabber.temp_calib())

    print(List[0])
    print(List[1])
    print(List[2])
    print(np.diff(np.array(List),n=2, axis=0)[0, 1])
    print(List[0][1] - List[2][1])


