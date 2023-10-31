import numpy as np
import matplotlib.pyplot as plt

import cv2


class TelemetryGrabber:
    channel_id_meaning = ("visible (0.58-0.68 µm)", "near-IR (0.725-1.0 µm)",
                          "Day near-IR (1.58-1.64 µm), Nigth infrared (3.55-3.93 µm)",
                          "infrared (10.3-11.3 µm)",
                          "infrared (11.5-12.5 µm)")
    telemetry_field_meaning = ("Zero Modulation Frame", "Thermistor Temp #1", "Thermistor Temp #2",
                               "Thermistor Temp #3", "Thermistor Temp #4", "Patch Temp", "Back Scan",
                               "Channel I.D. Wedge", "Wedge #1", "Wedge #2", "Wedge #3", "Wedge #4", "Wedge #5",
                               "Wedge #6", "Wedge #7", "Wedge #8")
    tel_positions = ((997, 1037), (2040, 2080))

    """
    path: a path to an image
    channel: A or B as string default A
    max_stdv_lines: the stdv within the lines of a telemetry frame to keep it counted default 4
    """

    def __init__(self, path, max_stdv_lines=4):
        self.img = cv2.imread(path, 0)
        self.max_stdv_lines = max_stdv_lines
        self.telemetry = [[], []]
        self.telemetry_raw = [[], []]

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
    Filter the telemetry
    """

    @staticmethod
    def filter(channel, img, max_stdv_lines):

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

        apt_frame = np.nanmedian(apt_frame_chunk, axis=(0, 2))
        return apt_frame, tel_raw_median



    def generate_telemetry(self):
        self.telemetry[0], self.telemetry_raw[0] = TelemetryGrabber.filter("A", self.img, self.max_stdv_lines)
        self.telemetry[1], self.telemetry_raw[1] = TelemetryGrabber.filter("B", self.img, self.max_stdv_lines)
        return self.telemetry, self.telemetry_raw

    def get_channel_type(self):
        if len(self.telemetry[0]) == 0:
            self.generate_telemetry()
        id_value_a = self.telemetry[0][7]
        id_value_b = self.telemetry[1][7]
        channel_a = TelemetryGrabber.find_nearest_index(self.telemetry[0][-8:], id_value_a)
        channel_b = TelemetryGrabber.find_nearest_index(self.telemetry[1][-8:], id_value_b)
        # !! The channel_a variable counts from 0 the numbering in online documentations counts from 1
        print(f"Channel A, {channel_a + 1}, contains these frequency's: {TelemetryGrabber.channel_id_meaning[channel_a]}")
        print(f"Channel B, {channel_b + 1}, contains these frequency's: {TelemetryGrabber.channel_id_meaning[channel_b]}")
        return channel_a, channel_b

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
