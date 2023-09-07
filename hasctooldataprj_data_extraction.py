import jax.numpy as jnp
import pandas as pd
from utils import find_closest_index
from main_constants import SIGNAL_LENGHT, HASC_FOLDER_PATH


def get_signal(folder_path, i):
    # Data creation

    # Read the CSV file without headers
    if i < 10:
        position_data = pd.read_csv(
            folder_path + "\\HASC100" + str(i) + ".csv", delimiter=",", header=None
        )
    else:
        position_data = pd.read_csv(
            folder_path + "\\HASC10" + str(i) + ".csv", delimiter=",", header=None
        )

    # Extract the columns based on their indices
    time = position_data.iloc[:, 0]
    x = position_data.iloc[:, 1]
    y = position_data.iloc[:, 2]
    z = position_data.iloc[:, 3]

    # Our signal avec the 3 mains variables x, y and z
    signal_1 = position_data.iloc[:, 1:4]

    # Read the segmentation file
    if i < 10:
        segmentation_data = pd.read_csv(
            folder_path + "\\HASC100" + str(i) + ".label",
            skiprows=1,
            header=None,
            names=["start", "end", "activity"],
        )
    else:
        segmentation_data = pd.read_csv(
            folder_path + "\\HASC10" + str(i) + ".label",
            skiprows=1,
            header=None,
            names=["start", "end", "activity"],
        )
    # Extract the start and end times for each segment
    start_times = segmentation_data["start"]
    end_times = segmentation_data["end"]
    activities = segmentation_data["activity"]
    # Create the ground truth segmentation :
    true_segmentation_signal_1 = []
    for i in range(len(start_times) - 1):
        true_segmentation_signal_1.append((end_times[i] + start_times[i + 1]) / 2)
    true_segmentation_signal_1.append(end_times[len(end_times) - 1])
    signal_1b = signal_1[
        0 : find_closest_index(time, true_segmentation_signal_1[-1:][0]) + 1
    ]

    true_segmentation_signal_1b = []
    for a in true_segmentation_signal_1:
        true_segmentation_signal_1b.append(find_closest_index(time, a))
    true_segmentation_signal_1b.append(len(signal_1b))

    return signal_1b, true_segmentation_signal_1b, time, activities


def load_hasc():
    print("Loading HascToolDataPrj database ...")
    nb_signals = 18
    # Data creation
    signals, segmentations, true_time_signals, true_activities = [], [], [], []
    for i in range(1, nb_signals + 1):
        signal, true_segmentation, true_time, activities = get_signal(
            HASC_FOLDER_PATH, i
        )
        signals.append(signal)
        segmentations.append(true_segmentation)
        true_time_signals.append(true_time)
        true_activities.append(activities)

    hasc_segmentations = []
    for segmentation in segmentations:
        filtered_segmentation = [
            element for element in segmentation if element <= SIGNAL_LENGHT
        ]
        if filtered_segmentation[len(filtered_segmentation) - 1] != SIGNAL_LENGHT:
            filtered_segmentation.append(SIGNAL_LENGHT)
        hasc_segmentations.append(jnp.array(filtered_segmentation))

    for i in range(len(true_activities)):
        true_activities[i] = true_activities[i][: len(hasc_segmentations[i])]

    hasc_signals = []
    for i in range(len(signals)):
        hasc_signals.append(signals[i][0:SIGNAL_LENGHT])
    print("Loading completed")
    print("Nb_signals : ", len(hasc_signals))
    return hasc_signals, hasc_segmentations
