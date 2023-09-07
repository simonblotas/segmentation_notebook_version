import numpy as np
import jax.numpy as jnp
from typing import Tuple
from typing import List, Generator
from six.moves import cPickle as pickle  # for performance


def find_closest_index(sorted_list: list[float], target_value: float) -> int:
    """
    Find the index of the element in a sorted array that is closest to a
    target value.

    Parameters:
    sorted_array (list[float]): A sorted list of values.
    target_value (float): The value for which the closest element's index is
    to be found.

    Returns:
    int: The index of the closest element to the target value in the
    sorted_array.
    """

    # Convert the sorted_array to a NumPy array for efficient computations
    sorted_array = np.asarray(sorted_list)

    # Use binary search to find the index where the target_value should be
    # inserted in sorted_array
    idx = np.searchsorted(sorted_array, target_value)

    # If target_value should be inserted at the beginning of sorted_array
    if idx == 0:
        return 0
    # If target_value should be inserted at the end of sorted_array
    elif idx == len(sorted_array):
        return len(sorted_array) - 1
    else:
        # Calculate the absolute differences between target_value and the
        # elements on both sides of idx
        left_diff = np.abs(sorted_array[idx - 1] - target_value)
        right_diff = np.abs(sorted_array[idx] - target_value)

        # Compare the absolute differences and choose the index with the
        # smallest difference
        if left_diff <= right_diff:
            return idx - 1
        else:
            return idx


def find_change_indices(array: jnp.ndarray) -> jnp.ndarray:
    """
    Finds indices where an array changes its values.

    Parameters:
    array (jnp.ndarray): The input array.

    Returns:
    jnp.ndarray: An array containing indices where the input array changes its values.
    """

    # Find indices where array changes its values
    change_indices = jnp.where(array[1:] != array[:-1])[0] + 1

    # Append the last index to account for the last change
    change_indices = jnp.append(change_indices, len(array))

    return change_indices


def create_data_loader(
    signals: jnp.ndarray,
    segmentations: jnp.ndarray,
    batch_size: int,
    test_batch_idx: List[int],
) -> Generator[Tuple[str, Tuple[jnp.ndarray, jnp.ndarray]], None, None]:
    """
    Creates a data loader for batched training data.

    Parameters:
    signals (jnp.ndarray): Array of signals.
    segmentations (jnp.ndarray): Array of segmentations.
    batch_size (int): Batch size for training.
    test_batch_idx (List[int]): List of batch indices to be used for testing.

    Yields:
    Tuple[str, Tuple[jnp.ndarray, jnp.ndarray]]: Batch type ("Train" or "Test") and data-target tuple.
    """
    num_samples = signals.shape[0]
    num_batches = num_samples // batch_size

    # Determine the start and end indices for each batch
    indices = jnp.arange(num_samples)

    # Split the indices into batches
    batch_indices = jnp.array_split(indices, num_batches)

    for batch_idx in range(num_batches):
        # Get the indices for the current batch
        batch_indices_curr = batch_indices[batch_idx]

        # Extract the signals and segmentations for the current batch
        data = signals[batch_indices_curr]
        target = segmentations[batch_indices_curr]

        if batch_idx in test_batch_idx:
            # Yield the test batch
            yield "Test", (data, target)
        else:
            # Yield the training batch
            yield "Train", (data, target)


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di
