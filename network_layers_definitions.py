import jax
import jax.lax as lax
from jax import random
import jax.numpy as jnp
import jax
from jax import lax
from jax import numpy as jnp


@jax.jit
def normalize_signal(signal):
    min_vals = jnp.min(signal, axis=0)
    max_vals = jnp.max(signal, axis=0)
    range_vals = max_vals - min_vals
    range_vals = jnp.where(range_vals < 1e-10, 1.0, range_vals)
    out = (signal - min_vals) / range_vals
    return out


def convolution_layer(kernel, bias, x, stride=2):
    """Simple convolutionnal layer"""
    dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ("NWC", "WIO", "NWC"))

    out_conv = lax.conv_general_dilated(
        x, kernel, (stride // 2,), "SAME", (1,), (1,), dn
    )

    return out_conv


def dense_layer(weights, bias, x):
    """Simple dense layer for single sample"""
    return jnp.dot(x, weights)


def initialize_linear_layer(m, n, key, scale=1):
    """Initialize weights for a linear (fully connected) layer"""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Function to initialize parameters with Gaussian (normal) distribution
def normal_initializer(shape, key, scale=1):
    return scale * jax.random.normal(key, shape)


def initialize(sizes, key):
    """Initialize the weights of all layers of a linear layer network"""
    keys = random.split(key, len(sizes))
    return [
        initialize_linear_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def initialize_network(
    parameters_informations, beta_initial=jnp.log(10.0), verbose=False
):
    linear_layer_sizes, conv_layer_sizes = (
        parameters_informations[0],
        parameters_informations[1],
    )
    key = random.PRNGKey(0)

    linear_params = [
        initialize(linear_layer, key) for linear_layer in linear_layer_sizes
    ]
    conv_params = [
        normal_initializer(kernel_size, key) for kernel_size in conv_layer_sizes
    ]

    # Create a dictionary 'params' to store all the parameters
    params = {}

    # Store linear parameters in the dictionary
    for i, layer_param in enumerate(linear_params):
        params[f"linear_layer_{i+1}_weights"] = layer_param[0][0]
        params[f"linear_layer_{i+1}_bias"] = layer_param[0][1]

    # Store convolutional parameters in the dictionary
    for i, layer_param in enumerate(conv_params):
        params[f"conv_layer_{i+1}_filter_weights"] = layer_param[0]
        params[f"conv_layer_{i+1}_bias"] = layer_param[1]

    # Store Beta parameter in the dictionary
    params[f"beta"] = jnp.array(beta_initial)

    if verbose:
        # Print the parameters in the 'params' dictionary
        print("Parameters:")
        for name, value in params.items():
            print(f"{name} - Shape: {value.shape}")

    return params
