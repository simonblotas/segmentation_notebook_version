import jax.numpy as jnp
from jax import jit
from jax import jit
from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
from typing import Dict, Callable, Tuple
from breakpoints_computation import get_optimal_projection, segmentation_to_projection


@jit
def compute_v_value(
    signal: jnp.ndarray, projection: jnp.ndarray, segmentation_size: int, beta: float
) -> float:
    """
    Computes the value of the function V for a given signal, projection, segmentation size, and penalty parameter.

    Parameters:
    signal (jnp.ndarray): The signal.
    projection (jnp.ndarray): The projection of the signal.
    segmentation_size (int): The size of the segmentation.
    beta (float): The penalty parameter.

    Returns:
    float: The computed value of the function V.
    """

    return ((signal - projection) ** 2).sum() + jnp.exp(beta) * segmentation_size


@jit
def loss(
    transformed_signal: jnp.ndarray, params: Dict, true_segmentation: jnp.ndarray
) -> float:
    """
    Computes the loss function for a given transformed signal, penalty
    parameter, and true segmentation.

    Parameters:
    transformed_signal (jnp.ndarray): The transformed signal.
    beta (float): The penalty parameter.
    true_segmentation (jnp.ndarray): The true segmentation points.

    Returns:
    float: The computed loss value.
    """
    # Calculate the projection and segment IDs using a prediction function
    pred_projection, pred_segmentation_size, segment_ids_pred = get_optimal_projection(
        transformed_signal, penalty=jnp.exp(params["beta"])
    )
    # Calculate the true projection and segmentation size
    true_projection = segmentation_to_projection(transformed_signal, true_segmentation)
    true_segmentation_size = true_segmentation[-1] + 1
    # Calculate the loss based on a difference in V values
    loss_value = (
        jnp.sum(
            compute_v_value(
                transformed_signal,
                true_projection,
                true_segmentation_size,
                params["beta"],
            )
            - compute_v_value(
                transformed_signal,
                pred_projection,
                pred_segmentation_size,
                params["beta"],
            )
        )
        / true_segmentation_size
    )
    return loss_value


def final_loss_and_grad(
    params: Dict,
    transformation: Callable,
    signals: jnp.ndarray,
    true_segmentation: jnp.ndarray,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """
    Compute the final loss and gradients for a transformation applied to a list of signals.

    Args:
        params (Dict): The parameters for the transformation.
        transformation (Callable): A callable transformation function.
        signals (jnp.ndarray): The input signals as a numpy array of shape (nb_signals, signal_length).
        true_segmentation (jnp.ndarray): The true segmentation data as a numpy array.

    Returns:
        Tuple[float, Dict[str, jnp.ndarray]]: A tuple containing:
            - final_loss (float): The final loss computed.
            - grads (Dict[str, jnp.ndarray]): Gradients with respect to parameters as a dictionary.
    """

    def main_loss(
        params: Dict,
        transformation: Callable,
        signal: jnp.ndarray,
        true_segmentation: jnp.ndarray,
    ) -> float:
        transformed_signal = transformation(params, signal)
        return loss(transformed_signal, params, true_segmentation)

    batched_value_and_grad = vmap(
        value_and_grad(main_loss, argnums=0, allow_int=True),
        in_axes=(
            None,
            None,
            0,
            0,
        ),
        out_axes=0,
    )
    losses, grads = batched_value_and_grad(
        params, transformation, signals, true_segmentation
    )
    final_loss = jnp.sum(losses)
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)

    return final_loss, grads
