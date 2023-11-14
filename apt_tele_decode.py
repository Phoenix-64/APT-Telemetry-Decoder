import numpy as np
import matplotlib.pyplot as plt
import cv2


class TelemetryGrabber:
    channel_id_meaning = ("Ch 1 visible (0.58-0.68 µm)", "Ch 2 near-IR (0.725-1.0 µm)",
                          "Ch 3A Day near-IR (1.58-1.64 µm)",
                          "Ch 4 infrared (10.3-11.3 µm)",
                          "Ch 5 infrared (11.5-12.5 µm)"
                          "Ch 3B Nigth infrared (3.55-3.93 µm)",)
    telemetry_field_meaning = ("Zero Modulation Frame", "Thermistor Temp #1", "Thermistor Temp #2",
                               "Thermistor Temp #3", "Thermistor Temp #4", "Patch Temp", "Back Scan",
                               "Channel I.D. Wedge", "Wedge #1", "Wedge #2", "Wedge #3", "Wedge #4", "Wedge #5",
                               "Wedge #6", "Wedge #7", "Wedge #8")
    # Radiation constants
    c1 = 1.1910427e-5
    c2 = 1.4387752
    # Conversion Coefficients Noaa 15, 18, 19
    ds = np.array([[[276.60157, 0.051045, 1.36328E-06],
                    [276.62531, 0.050909, 1.47266E-06],
                    [276.67413, 0.050907, 1.47656E-06],
                    [276.59258, 0.050966, 1.47656E-06]],

                   [[276.601, 0.05090, 1.657e-06],
                    [276.683, 0.05101, 1.482e-06],
                    [276.565, 0.05117, 1.313e-06],
                    [276.615, 0.05103, 1.484e-06]],

                   [[276.6067, 0.051111, 1.405783E-06],
                    [276.6119, 0.051090, 1.496037E-06],
                    [276.6311, 0.051033, 1.496990E-06],
                    [276.6268, 0.051058, 1.493110E-06]]])
    # Temperature-to-radiance coefficients for NOAA-15, 18, 19 Ch 3b Ch 4 Ch 5  in each vc A B
    ttrc = np.array([[[2669.3554, 1.702380, 0.997378],
                      [926.2947, 0.271683, 0.998794],
                      [839.8246, 0.309180, 0.999012]],
                     [[2659.7952, 1.698704, 0.996960],
                      [928.1460, 0.436645, 0.998607],
                      [833.2532, 0.253179, 0.999057]],
                     [[2670.0, 1.67396, 0.997364],
                      [928.9, 0.53959, 0.998534],
                      [831.9, 0.36064, 0.998913]]])
    # Radiance of Space and Coefficients for Nonlinear Radiance Correction Quadratic. Noaa 15, 18, 19 Ch4 Ch5, Ns b0 b1 b2
    RsNRC = np.array([[[-4.50, 4.76, -0.0932, 0.0004524],
                       [-3.61, 3.83, -0.0659, 0.0002811]],
                      [[-5.53, 5.82, -0.11069, 0.00052337],
                       [-2.22, 2.67, -0.04360, 0.00017715]],
                      [[-5.49, 5.70, -0.11187, 0.00054668],
                       [-3.39, 3.58, -0.05991, 0.00024985]]])
    # all possible earth counts:
    Ce = np.arange(0, 256)

    img_pos = ((88, 992), (1128, 2033))
    tel_positions = ((997, 1037), (2040, 2080))
    space_position = ((40, 84), (1084, 1124))
    calib_value_arrangement = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    lut_out = [0, 31, 63, 95, 127, 159, 191, 223, 255]

    """
    path: a path to an image
    channel: A or B as string default A
    max_stdv_lines: the stdv within the lines of a telemetry frame to keep it counted default 4
    satellite takes eiter 0, 1 or 2 for NOAA 15, 18, 19
    """

    def __init__(self, path, satellite=np.nan, max_stdv_lines=4):
        self.img = cv2.imread(path, 0)
        self.max_stdv_lines = max_stdv_lines
        self.satellite = satellite
        self.telemetry = [[], []]
        self.telemetry_raw = [[], []]
        self.space = [[], []]
        self.Te = [[], []]

    @staticmethod
    def get_median_strip(data, x1, x2):
        tel = data[0:data.shape[0], x1:x2]
        tel_mean = np.median(tel, axis=1)
        return tel_mean

    @staticmethod
    def find_runs(x):
        """Find runs of consecutive items in an array."""

        # ensure array
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # find run values
            run_values = x[loc_run_start]

            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            return run_values, run_starts, run_lengths

    @staticmethod
    def cut_to_chunks(data, start, chunk_len):
        first_index = start - (start // chunk_len * chunk_len)
        last_index = start + ((len(data) - start) // chunk_len * chunk_len) - chunk_len
        nr_blocks = (last_index - first_index) // chunk_len

        return first_index, nr_blocks

    @staticmethod
    def find_nearest_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def get_telemetry_meaning(field):
        print(f"This field describes the following: {TelemetryGrabber.telemetry_field_meaning[field]}")

    """
    Filter the Space Data
    """

    @staticmethod
    def space_filter(channel, img):
        if channel == "A":
            space_pos = TelemetryGrabber.space_position[0]
        else:
            space_pos = TelemetryGrabber.space_position[1]

        space_raw_median = TelemetryGrabber.get_median_strip(img, space_pos[0], space_pos[1])[:-1]

        hist = np.histogram(space_raw_median)
        # if more are in lower end then black is default
        if hist[0][0] > hist[0][-1]:
            mask = np.digitize(space_raw_median, hist[1][:2])
            space_raw_median[mask > 1] = np.nan
        else:
            mask = np.digitize(space_raw_median, hist[1][-2:])
            space_raw_median[mask < 1] = np.nan

        # then replace all in mask with nan and take median

        space = np.nanmedian(space_raw_median)
        return space

    """
    Filter the telemetry
    """

    @staticmethod
    def tel_filter(channel, img, max_stdv_lines):
        if channel == "A":
            tel_pos = TelemetryGrabber.tel_positions[0]
        else:
            tel_pos = TelemetryGrabber.tel_positions[1]

        tel_raw_median = TelemetryGrabber.get_median_strip(img, tel_pos[0], tel_pos[1])[:-1]

        # set delta low rais until we get 8 consecutive numbers then find index of first
        delta = 2
        while True:
            below_threshold = np.flatnonzero(tel_raw_median < (tel_raw_median.min() + delta))
            difference = np.diff(below_threshold)
            if len(difference) == 0:
                delta += 0.1
                continue
            runs = TelemetryGrabber.find_runs(difference)
            if runs[2].max() >= 7:
                break
            delta += 0.1

        # break up the 8 lines that correspond to 1 frame into another dimension
        starting_index_run = below_threshold[runs[1][np.argmax(runs[2])]]
        first_index, nr_blocks = TelemetryGrabber.cut_to_chunks(tel_raw_median, starting_index_run, 8)
        tel_frame_chunk = np.zeros(shape=(nr_blocks, 8))
        for i in range(0, nr_blocks):
            tel_frame_chunk[i] = tel_raw_median[first_index + i * 8:first_index + (i + 1) * 8]

        # take median of all 8 lines per frame and find the one with
        # the lowest value next to the highest for orientation
        tel_frame = np.median(tel_frame_chunk, axis=1)
        zero_frame = np.argmin(np.diff(tel_frame)) + 1

        # then create an n by 16 by 8 apt frame construction as follows:  atp_frame, telemetry_frame, line
        first_index, nr_blocks = TelemetryGrabber.cut_to_chunks(tel_frame, zero_frame, 16)
        apt_frame_chunk = np.zeros(shape=(nr_blocks, 16, 8))
        for i in range(0, nr_blocks):
            apt_frame_chunk[i] = tel_frame_chunk[first_index + i * 16:first_index + (i + 1) * 16]

        # only keep telemetry frames with low stdv
        std = np.std(apt_frame_chunk, axis=2)
        for idi, i in enumerate(std):  # idi 3 max
            for idj, j in enumerate(i):  # idj 15 max
                if j > max_stdv_lines:
                    apt_frame_chunk[idi, idj] = np.nan

        apt_frame = np.roll(np.nanmedian(apt_frame_chunk, axis=(0, 2)), 8)
        return apt_frame, tel_raw_median

    def generate_telemetry(self):
        self.telemetry[0], self.telemetry_raw[0] = TelemetryGrabber.tel_filter("A", self.img, self.max_stdv_lines)
        self.telemetry[1], self.telemetry_raw[1] = TelemetryGrabber.tel_filter("B", self.img, self.max_stdv_lines)
        return self.telemetry, self.telemetry_raw

    def generate_space(self):
        self.space[0] = TelemetryGrabber.space_filter("A", self.img)
        self.space[1] = TelemetryGrabber.space_filter("B", self.img)
        return self.space

    """
    This is used to find the instrument channel transmitted
    """

    def get_channel_type(self):
        if len(self.telemetry[0]) == 0:
            self.generate_telemetry()
        id_value_a = self.telemetry[0][15]
        id_value_b = self.telemetry[1][15]
        channel_a = TelemetryGrabber.find_nearest_index(self.telemetry[0][:8], id_value_a)
        channel_b = TelemetryGrabber.find_nearest_index(self.telemetry[1][:8], id_value_b)
        # !! The channel_a variable counts from 0 the numbering in online documentations counts from 1
        print(
            f"Channel A, {channel_a + 1}, contains these frequency's: {TelemetryGrabber.channel_id_meaning[channel_a]}")
        print(
            f"Channel B, {channel_b + 1}, contains these frequency's: {TelemetryGrabber.channel_id_meaning[channel_b]}")
        return channel_a, channel_b

    """
    This function matches the wedges 1-8 to there ideal value 
    """

    def correct_img_lvls(self, lut=lut_out):
        # merged both telemetry frames to optimize for whole image
        if len(self.telemetry[0]) == 0:
            self.generate_telemetry()
        telemetry = np.mean(self.telemetry, axis=0)
        lut_in = []
        for i in TelemetryGrabber.calib_value_arrangement:
            lut_in.append(telemetry[i])
        lut_8u = np.interp(np.arange(0, 256), lut_in, lut).astype(np.uint8)
        self.img = cv2.LUT(self.img, lut_8u)
        # call telemetry generator to update
        self.generate_telemetry()
        self.generate_space()
        return self.img

    """
    This generates the temperature calibration data for both channels 
    """

    def temp_calib(self):
        if np.isnan(self.satellite):
            print("You first need to define what satellite this image stems from, a.satellite")
            return False
        Rs = np.zeros(4)
        channels = self.get_channel_type()
        ds = TelemetryGrabber.ds[self.satellite]
        ttrc = TelemetryGrabber.ttrc[self.satellite]
        RsNRC = TelemetryGrabber.RsNRC[self.satellite]
        self.correct_img_lvls()
        Cprts = np.array(self.telemetry)[:, 9:13] * 4
        Cbb = np.array(self.telemetry)[:, 13] * 4
        Cs = np.array(self.space) * 4

        Tprts = np.zeros(shape=Cprts.shape)
        for j in range(0, 2):
            for i in range(0, 4):
                Tprts[j][i] = ds[i][0] + ds[i][1] * Cprts[j][i] + ds[i][2] * ((Cprts[j][i]) ** 2)

        Tbb = np.median(Tprts, axis=1)
        Tw14 = 0.124 * np.array(self.telemetry)[:, 14] + 90.113

        Tbbs = np.zeros(shape=2)
        Nbb = np.zeros(shape=2)
        Nlin = np.zeros(shape=(2, len(TelemetryGrabber.Ce)))
        Ncor = np.zeros(shape=Nlin.shape)
        Ne = np.zeros(shape=Nlin.shape)
        Tes = np.zeros(shape=Nlin.shape)
        Te = np.zeros(shape=Nlin.shape)
        for i in range(0, 2):
            if channels[i] == 3:
                Rs = RsNRC[0]
                ttrcl = ttrc[1]
            elif channels[i] == 4:
                Rs = RsNRC[1]
                ttrcl = ttrc[2]
            elif channels[i] == 5:
                ttrcl = ttrc[0]
            else:
                continue
            Tbbs[i] = ttrcl[1] + ttrcl[2] * Tbb[i]
            Nbb[i] = ((TelemetryGrabber.c1 * (ttrcl[0] ** 3))
                      / ((np.e ** ((TelemetryGrabber.c2 * ttrcl[0]) / Tbbs[i])) - 1))
            Nlin[i] = Rs[0] + (Nbb[i] - Rs[0]) * ((Cs[i] - (TelemetryGrabber.Ce * 4)) / (Cs[i] - Cbb[i]))
            Ncor[i] = Rs[1] + Rs[2] * Nlin[i] + Rs[3] * (Nlin[i] ** 2)
            Ne[i] = Nlin[i] + Ncor[i]

            Tes[i] = (TelemetryGrabber.c2 * ttrcl[0]) / np.log(1 + ((TelemetryGrabber.c1 * (ttrcl[0] ** 3)) / Ne[i]))
            Te[i] = (Tes[i] - ttrcl[1]) / ttrcl[2]
        self.Te = [[], []]
        return Te

    """
    Create a false color image of the Temperature 
    """

    def falsecolor_temp(self, cmap="jet", filter_size=6, bars=False):
        maps = []
        temp_imgs = []
        img_pos = TelemetryGrabber.img_pos
        Te = self.temp_calib()

        # Filter Te for nan values
        for idx, x in enumerate(Te[1]):
            if np.isnan(x):
                Te[1, idx] = Te[1, idx - 1]
            else:
                Te[1, idx] = x


        img_b = self.img[:, img_pos[1][0]:img_pos[1][1]]
        # x size equal to wxtoimg for direct value comp
        #img_b = self.img[:, 1039:]

        temp_img_b = Te[1][img_b] - 273.15
        temp_filtered_b = cv2.bilateralFilter(temp_img_b.astype(np.float32), filter_size, 75, 75)
        temp_imgs.append(temp_img_b)
        temp_imgs.append(temp_filtered_b)



        if np.mean(Te[0]) > 3:
            img_a = self.img[:, img_pos[0][0]:img_pos[0][1]]
            temp_img_a = Te[0][img_a] - 273.15
            temp_filtered_a = cv2.bilateralFilter(temp_img_a.astype(np.float32), filter_size, 75, 75)
            temp_imgs.append(temp_img_a)
            temp_imgs.append(temp_filtered_a)

            fig, ax = plt.subplots(2, 2)
            maps.append(ax[0, 0].imshow(temp_img_b, cmap=cmap))
            maps.append(ax[0, 1].imshow(temp_filtered_b, cmap=cmap))
            maps.append(ax[1, 0].imshow(temp_img_a, cmap=cmap))
            maps.append(ax[1, 1].imshow(temp_filtered_a, cmap=cmap))

        else:
            fig, ax = plt.subplots(1, 2)
            maps.append(ax[0].imshow(temp_img_b, cmap=cmap))
            maps.append(ax[1].imshow(temp_filtered_b, cmap=cmap))

        if bars:
            fig.colorbar(maps[-1])
        fig.tight_layout()
        plt.show()
        return temp_imgs

    """
    To get a plot of the apt telemetry frame that was decoded
    """

    def visualize_telemetry(self):
        if len(self.telemetry[0]) == 0:
            self.generate_telemetry()

        fig, ((ax, ax2), (ax1, ax3)) = plt.subplots(2, 2)
        ax.imshow(np.atleast_2d(self.telemetry[0]), cmap=plt.get_cmap('gray'),
                  extent=(0, len(self.telemetry[0]), 0, len(self.telemetry[0]) / 4))
        ax1.imshow(np.atleast_2d(self.telemetry[1]), cmap=plt.get_cmap('gray'),
                   extent=(0, len(self.telemetry[1]), 0, len(self.telemetry[1]) / 4))
        ax2.imshow(np.atleast_2d(self.telemetry_raw[0]), cmap=plt.get_cmap('gray'),
                   extent=(0, len(self.telemetry_raw[0]), 0, len(self.telemetry_raw[0]) / 4))
        ax3.imshow(np.atleast_2d(self.telemetry_raw[1]), cmap=plt.get_cmap('gray'),
                   extent=(0, len(self.telemetry_raw[1]), 0, len(self.telemetry_raw[1]) / 4))
        plt.show()
