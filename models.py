from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import numpy as np
import optax
import ruptures as rpt
import jax.numpy as jnp
from jax import vmap
import time
from breakpoints_computation import get_optimal_projection
from utils import find_change_indices, create_data_loader
from loss_related_functions import final_loss_and_grad
from default_optimizer import gradient_transform
from ruptures.metrics.precisionrecall import precision_recall


class SimplePipeline(BaseEstimator):
    _required_parameters = ["estimator"]

    def __init__(
        self,
        transformation_method,
        initialize_parameters_method,
        parameters_informations,
        optimizer=gradient_transform,
    ):
        self.parameters_informations = parameters_informations
        self.initialize_parameters_method = initialize_parameters_method
        self.params = self.initialize_parameters_method(
            self.parameters_informations, verbose=True
        )
        self.transformation_method = transformation_method
        self.optimizer = optimizer

    def fit(
        self,
        signals,
        segmentations,
        verbose=False,
        num_epochs=10,
        batch_size=5,
        test_batch_idx=[],
        **fit_params
    ):
        """Implements a learning loop over epochs."""

        self.params = self.initialize_parameters_method(self.parameters_informations)
        # Initialize placeholder fit data
        train_loss = []
        acc_train = []
        acc_test = []
        # Initialize optimizer.
        opt_state = self.optimizer.init(self.params)
        # Loop over the training epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loader = create_data_loader(
                signals, segmentations, batch_size, test_batch_idx
            )
            bacth_losses = []
            batch_acc_train = []
            batch_acc_test = []
            for batch_type, (data, target) in train_loader:
                if batch_type == "Train":
                    data = jnp.array(data)
                    target = jnp.array(target)
                    value, grads = final_loss_and_grad(
                        self.params, self.transformation_method, data, target
                    )
                    updates, opt_state = self.optimizer.update(grads, opt_state)
                    self.params = optax.apply_updates(self.params, updates)
                    bacth_losses.append(value)

                elif batch_type == "Test":
                    pass

            epoch_loss = np.mean(bacth_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time
            if verbose:
                print(
                    "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                        epoch + 1,
                        epoch_time,
                        epoch_acc_train,
                        epoch_acc_test,
                        epoch_loss,
                        jnp.exp(self.params["beta"]),
                    )
                )

        return train_loss, opt_state, self.params

    def predict(self, signals):
        def predict_segmentation(self, signal: jnp.ndarray) -> jnp.ndarray:
            """
            Predicts the segmentation of a given signal using the trained scaling network.

            Parameters:
            signal (jnp.ndarray): The input signal for segmentation prediction.

            Returns:
            jnp.ndarray: Predicted segmentation indices for the input signal.
            """
            # Transform the input signal using the network's transformation function
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signal)
            )

            # Convert the transformed signal to a numpy array for compatibility
            # transformed_signal_array = np.array(transformed_signal)

            # Get the optimal projection and predicted segmentation using the transformed signal
            (
                pred_projection,
                pred_segmentation_size,
                segment_ids_pred,
            ) = get_optimal_projection(
                transformed_signal, penalty=jnp.exp(self.params["beta"])
            )
            # print(segment_ids_pred[1])
            # Find and return the predicted segmentation indices
            # predicted_segmentation = find_change_indices(segment_ids_pred[1])
            return segment_ids_pred[1]
            # Make a batched version of the `predict_segmentation` function

        jnp_predictions = vmap(predict_segmentation, in_axes=(None, 0), out_axes=0)(
            self, signals
        )
        predictions = [
            find_change_indices(jnp_predictions[i]) for i in range(len(jnp_predictions))
        ]

        return predictions

    def display(self, signals, true_segmentations):
        predictions = self.predict(signals)
        print("marge = ", signals.shape[1] * (5 / 100) )
        for i in range(len(signals)):
            transformed_signal = self.transformation_method(
                self.params, jnp.array(signals[i])
            )
            precision, recall = precision_recall(
            np.array(predictions[i]),
            np.array(find_change_indices(true_segmentations[i])),
            margin=signals.shape[1] * (5 / 100),
            )
            print(predictions[i])
            print(find_change_indices(true_segmentations[i]))
            if precision + recall == 0:
                print(0)
            else:
                print(2 * (precision * recall) / (precision + recall))
            rpt.display(
                np.array(transformed_signal),
                find_change_indices(true_segmentations[i]),
                predictions[i],
            )
            plt.show()
