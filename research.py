import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("raw.png", 0)
tel_pos = (2040, 2080)
# the maximum stdv that the lines can have to be counted as a good telemetry frame.
max_stdv_lines = 4
telemetry_field_meaning = ("Zero Modulation Frame", "Thermistor Temp #1", "Thermistor Temp #2", "Thermistor Temp #3",
                           "Thermistor Temp #4", "Patch Temp", "Back Scan", "Channel I.D. Wedge", "Wedge #1",
                           "Wedge #2", "Wedge #3", "Wedge #4", "Wedge #5", "Wedge #6", "Wedge #7", "Wedge #8")
channel_id_wedge_meaning = ("visible (0.58-0.68 µm)", "near-IR (0.725-1.0 µm)",
                            "Day near-IR (1.58-1.64 µm), Nigth infrared (3.55-3.93 µm)", "infrared (10.3-11.3 µm)",
                            "infrared (11.5-12.5 µm)")


def get_mean_strip(data, x1, x2):
    tel = data[0:data.shape[0], x1:x2]
    tel_mean = np.median(tel, axis=1)
    return tel_mean


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


def cut_to_chunks(data, start, l):
    first_index = start - (start // l * l)
    last_index = start + ((len(data) - start) // l * l) - l
    nr_blocks = (last_index - first_index) // l

    return first_index, nr_blocks


b_tel = img[0:img.shape[0], 2040:2080][:-1]
# get average of strip
b_tel_mean = get_mean_strip(img, tel_pos[0], tel_pos[1])[:-1]


# set delta low rais until we get 8 conescutiv numbers then find index of first
delta = 2
while True:
    below_threshold = np.flatnonzero(b_tel_mean < (b_tel_mean.min() + delta))
    diffrence = np.diff(below_threshold)
    runs = find_runs(diffrence)
    if runs[2].max() >= 7:
        break
    delta += 0.1

# brack up the 8 lines that corespond to 1 frame into another dimension
starting_index_run = below_threshold[runs[1][np.argmax(runs[2])]]
first_index, nr_blocks = cut_to_chunks(b_tel_mean, starting_index_run, 8)
tel_frame_chunk = np.zeros(shape=(nr_blocks, 8))
for i in range(0, nr_blocks):
    tel_frame_chunk[i] = b_tel_mean[first_index + i * 8:first_index + (i + 1) * 8]

# take midean of all 8 lines per frame and find the one with the lowest value for orentiation
tel_frame = np.median(tel_frame_chunk, axis=1)
zero_frame = np.argmin(tel_frame)

# then create a n by 16 by 8 apt frame construction as follows:  atp_frame, telematry_frame, line
first_index, nr_blocks = cut_to_chunks(tel_frame, zero_frame, 16)

apt_frame_chunk = np.zeros(shape=(nr_blocks, 16, 8))
for i in range(0, nr_blocks):
    apt_frame_chunk[i] = tel_frame_chunk[first_index + i * 16:first_index + (i + 1) * 16]

apt_frame_dirty = np.median(apt_frame_chunk, axis=(0, 2))

# only keep telemetry frames with low stdv
std = np.std(apt_frame_chunk, axis=2)
for idi, i in enumerate(std):  # idi 3 max
    for idj, j in enumerate(i):  # idj 15 max
        if j > max_stdv_lines:
            apt_frame_chunk[idi, idj] = np.nan

apt_frame = np.nanmedian(apt_frame_chunk, axis=(0, 2))

fig, (ax, ax1, ax2, ax3) = plt.subplots(4)
ax.imshow(np.atleast_2d(apt_frame_dirty), cmap=plt.get_cmap('gray'), extent=(0, len(tel_frame), 0, len(tel_frame) / 4))
ax1.imshow(np.atleast_2d(apt_frame), cmap=plt.get_cmap('gray'), extent=(0, len(tel_frame), 0, len(tel_frame) / 4))
ax2.imshow(tel_frame_chunk.T, cmap=plt.get_cmap('gray'))
ax3.imshow(np.atleast_2d(tel_frame), cmap=plt.get_cmap('gray'), extent=(0, len(tel_frame), 0, len(tel_frame) / 4))
plt.show()
