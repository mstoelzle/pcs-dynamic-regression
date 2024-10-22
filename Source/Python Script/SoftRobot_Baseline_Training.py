import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras
from pathlib import Path

# dataset parameters
dataset_dir = Path("Source") / "Soft Robot" / "ns-1_high_shear_stiffness" / "training" / "cv"
val_ratio = 0.2
# model parameters
model_type = "node"
# training parameters
lr = 1e-3
batch_size = 128
epochs = 500


if __name__ == "__main__":
    # Load the data
    Y, Y_d = np.load(dataset_dir / "Y.npy"), np.load(dataset_dir / "Ydot.npy")
    Tau = np.load(dataset_dir / "Tau.npy")
    num_samples = Y.shape[0]
    n_chi = Y.shape[-1] // 2
    n_tau = Tau.shape[-1]
    Chi, Chi_d = Y[:, :n_chi], Y[:, n_chi:]
    Chi_dd = Y_d[:, n_chi:]
    # reshape Tau to match the input shape
    Tau = Tau.reshape(num_samples, n_tau)
    print("Chi shape:", Chi.shape, "Chi_d shape:", Chi_d.shape, "Chi_dd shape:", Chi_dd.shape)
    print("Tau shape:", Tau.shape)
    print("Chi min", Chi.min(), "Chi max", Chi.max(), )

    # Split the data into training and validation sets
    # num_val_samples = int(num_samples * val_ratio)
    # num_train_samples = num_samples - num_val_samples
    # print("Number of training samples:", num_train_samples, "Number of validation samples:", num_val_samples)
    # Chi_train, Chi_val = Chi[:num_train_samples], Chi[num_train_samples:]
    # Chi_d_train, Chi_d_val = Chi_d[:num_train_samples], Chi_d[num_train_samples:]
    # Chi_dd_train, Chi_dd_val = Chi_dd[:num_train_samples], Chi_dd[num_train_samples:]
    # Tau_train, Tau_val = Tau[:num_train_samples], Tau[num_train_samples:]

    # construct the input and output data
    # x_train = np.concat((Chi_train, Chi_d_train, Tau_train), axis=-1)
    # y_train = Chi_dd_train
    # x_val = np.concat((Chi_val, Chi_d_val, Tau_val), axis=-1)
    # y_val = Chi_dd_val
    x = np.concatenate((Chi, Chi_d, Tau), axis=-1)
    y = Chi_dd

    # normalize the input data
    input_normalization_layer = keras.layers.Normalization(axis=-1)
    input_normalization_layer.adapt(x)
    print("Input normalization layer mean:", input_normalization_layer.mean.mean())
    # normalize the output data
    output_normalization_layer = keras.layers.Normalization(axis=-1)
    output_normalization_layer.adapt(y)
    y_norm = output_normalization_layer(y)
    print("Mean of output data:", y_norm.mean(axis=0))

    match model_type:
        case "node":
            input_dim = 2 * n_chi + n_tau
            output_dim = n_chi

            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(input_dim, )),
                    input_normalization_layer,
                    keras.layers.Dense(32, activation="tanh"),
                    keras.layers.Dense(32, activation="tanh"),
                    keras.layers.Dense(output_dim),
                ]
            )
        case _:
            raise ValueError("Invalid model type")
    model.summary()

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
        ],
    )

    callbacks = [
        # keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]

    model.fit(
        x,
        y_norm,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_ratio,
        callbacks=callbacks,
    )
