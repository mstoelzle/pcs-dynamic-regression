import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import jax.numpy as jnp
import keras
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

from baseline_dynamical_models import ConDynamics, LnnDynamics, generate_positive_definite_matrix_from_params


# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(0)


# dataset parameters
dataset_dir = Path("Source") / "Soft Robot" / "ns-2_dof-3" / "training" / "cv"
val_ratio = 0.2
# model parameters
model_type = "node"
assert model_type in ["node", "con", "lnn"]
mlp_num_layers = 5
mlp_hidden_dim = 256
# training parameters
lr = 1e-3
batch_size = 256
num_epochs = 2000

# directory to save the model
model_dir = dataset_dir.parent.parent / "model"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / f"learned_{model_type}_model.keras"
norm_model_path = model_dir / f"learned_{model_type}_model_norm.keras"


if __name__ == "__main__":
    # Load the data
    X = np.load(dataset_dir / "X.npy")  # configuration-space data
    Chi_raw = np.load(dataset_dir / "Chi_raw.npy")  # the raw poses in Cartesian space
    Y, Y_d = np.load(dataset_dir / "Y.npy"), np.load(dataset_dir / "Ydot.npy")
    Tau = np.load(dataset_dir / "Tau.npy")
    num_samples = Y.shape[0]
    num_videos = X.shape[0]
    num_samples_per_video = num_samples // num_videos
    num_markers = Chi_raw.shape[1] // 3
    n_chi = Y.shape[-1] // 2
    n_tau = Tau.shape[-1]
    Chi, Chi_d = Y[:, :n_chi], Y[:, n_chi:]
    Chi_dd = Y_d[:, n_chi:]
    # reshape Tau to match the input shape
    Tau = Tau.reshape(num_samples, n_tau)

    dt = 1e-3
    ts = dt * np.arange(num_samples)

    # # load the configuration-space data
    # X_copy = np.load(dataset_dir / "X_copy2.npy")
    # # plot the configuration-space data
    # fix, axes = plt.subplots(1, 2, figsize=(10, 5))
    # points = np.arange(0, X.shape[1])
    # print("Points:", points.shape)
    # print("X shape:", X.shape, "X_copy shape:", X_copy.shape)
    # print("mean X:", X.mean(), "mean X_copy:", X_copy.mean())
    # X_min, X_max = X.min(), X.max()
    # for i in range(12):
    #     axes[0].plot(points, X[0, :, i])
    #     axes[1].plot(points, X_copy[0, :, i])
    # axes[0].set_title("Configurations from dynamics repo")
    # axes[1].set_title("Configurations from kinematics repo")
    # axes[0].set_ylim(X_min, X_max)
    # axes[1].set_ylim(X_min, X_max)
    # axes[0].grid(True)
    # axes[1].grid(True)
    # plt.show()

    # marker sub-sampling
    # print("Number of markers:", num_markers)
    # marker_indices = jnp.array([num_markers - 1])
    # marker_indices = jnp.array([num_markers // 2, num_markers - 1])
    marker_indices = jnp.array([8, 14, 20])
    print("Selected marker indices:", marker_indices)
    # reshape tensors
    Chi = Chi.reshape(num_samples, num_markers, 3)
    Chi_raw = Chi_raw.reshape(num_samples, num_markers, 3)
    Chi_d = Chi_d.reshape(num_samples, num_markers, 3)
    Chi_dd = Chi_dd.reshape(num_samples, num_markers, 3)
    # sub-sample the data
    Chi_raw = Chi_raw[:, marker_indices, :].reshape(num_samples, -1)
    Chi = Chi[:, marker_indices, :].reshape(num_samples, -1)
    Chi_d = Chi_d[:, marker_indices, :].reshape(num_samples, -1)
    Chi_dd = Chi_dd[:, marker_indices, :].reshape(num_samples, -1)
    # update the number of markers
    num_markers = marker_indices.shape[0]
    print("Number of markers:", num_markers)
    n_chi = Chi.shape[-1]

    # # smooth the data
    # Chi, Chi_d, Chi_dd = [], [], []
    # for i in range(num_videos):
    #     savgol_window_length = 51
    #     Chi_i_raw = Chi_raw[i * num_samples_per_video:(i + 1) * num_samples_per_video]
    #     Chi_i = savgol_filter(Chi_i_raw, window_length=savgol_window_length, polyorder=3, deriv=0, delta=dt, axis=0)
    #     Chi_d_i = savgol_filter(Chi_i_raw, window_length=savgol_window_length, polyorder=3, deriv=1, delta=dt, axis=0)
    #     Chi_dd_i = savgol_filter(Chi_i_raw, window_length=savgol_window_length, polyorder=3, deriv=2, delta=dt, axis=0)
    #     Chi.append(Chi_i)
    #     Chi_d.append(Chi_d_i)
    #     Chi_dd.append(Chi_dd_i)
    # Chi, Chi_d, Chi_dd = np.concatenate(Chi, axis=0), np.concatenate(Chi_d, axis=0), np.concatenate(Chi_dd, axis=0)
      
    # plot the end effector data
    plt.figure(num="End effector pose")
    plt.plot(ts, Chi_raw[:, -3], linestyle=":", linewidth=2.5, label=r"$x$")
    plt.plot(ts, Chi_raw[:, -2], linestyle=":", linewidth=2.5, label=r"$y$")
    plt.plot(ts, Chi_raw[:, -1], linestyle=":", linewidth=2.5, label=r"$\theta$")
    # reset the color cycle
    plt.gca().set_prop_cycle(None)
    plt.plot(ts, Chi[:, -3], label=r"$\hat{x}$")
    plt.plot(ts, Chi[:, -2], label=r"$\hat{y}$")
    plt.plot(ts, Chi[:, -1], label=r"$\hat{\theta}$")
    plt.xlabel("Time [s]")
    plt.ylabel("End effector pose")
    plt.grid(True)
    plt.legend()
    plt.show()

    # # plot the pose data
    # plt.figure(num="Pose: x-coordinates")
    
    # for i in range(0, Chi.shape[1], 3):
    #     plt.plot(ts, Chi[:, i], label=f"Segment {i // 3 + 1}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("x-coordinate [m]")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(num="Pose: y-coordinates")
    # for i in range(1, Chi.shape[1], 3):
    #     plt.plot(ts, Chi[:, i], label=f"Segment {i // 3 + 1}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("y-coordinate [m]")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(num="Pose: orientations")
    # for i in range(2, Chi.shape[1], 3):
    #     plt.plot(ts, Chi[:, i], label=f"Segment {i // 3 + 1}")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Orientation [rad]")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plot acceleration data
    plt.figure(num="Acceleration: x-coordinates")
    for i in range(0, Chi_dd.shape[1], 3):
        plt.plot(ts, Chi_dd[:, i], label=fr"Marker {i // 3 + 1}")
    plt.xlabel("Time [s]")
    plt.ylabel(r"x-acceleration [m/s$^2$]")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(num="Acceleration: y-coordinates")
    for i in range(1, Chi_dd.shape[1], 3):
        plt.plot(ts, Chi_dd[:, i], label=fr"Marker {i // 3 + 1}")
    plt.xlabel("Time [s]")
    plt.ylabel(r"y-acceleration [m/s$^2$]")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(num="Acceleration: orientations")
    for i in range(2, Chi_dd.shape[1], 3):
        plt.plot(ts, Chi_dd[:, i], label=rf"Marker {i // 3 + 1}")
    plt.xlabel("Time [s]")
    plt.ylabel(r"Angular acceleration [rad/s$^2$]")
    plt.grid(True)
    plt.legend()
    plt.show()

    # plot the torque data
    plt.figure(num="Torque")
    for i in range(Tau.shape[1]):
        plt.plot(ts, Tau[:, i], label=fr"Torque $\tau_{i + 1}$")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Split the data into training and validation sets
    num_val_samples = int(num_samples * val_ratio)
    num_train_samples = num_samples - num_val_samples
    print("Number of training samples:", num_train_samples, "Number of validation samples:", num_val_samples)
    Chi_train, Chi_val = Chi[:num_train_samples], Chi[num_train_samples:]
    Chi_d_train, Chi_d_val = Chi_d[:num_train_samples], Chi_d[num_train_samples:]
    Chi_dd_train, Chi_dd_val = Chi_dd[:num_train_samples], Chi_dd[num_train_samples:]
    Tau_train, Tau_val = Tau[:num_train_samples], Tau[num_train_samples:]

    # construct the input and output data
    x = np.concat((Chi, Chi_d, Tau), axis=-1)
    y = Chi_dd
    x_train = np.concat((Chi_train, Chi_d_train, Tau_train), axis=-1)
    y_train = Chi_dd_train
    x_val = np.concat((Chi_val, Chi_d_val, Tau_val), axis=-1)
    y_val = Chi_dd_val

    # normalize the input data
    input_normalization_layer = keras.layers.Normalization(axis=-1)
    input_normalization_layer.adapt(x)


    # normalize the output data
    chi_i_dd_norm_const = np.array([500, 500, 25000])
    chi_dd_norm_const = np.tile(chi_i_dd_norm_const, num_markers)
    normalization_kwargs = dict(
        axis=-1,
        mean=np.zeros(chi_dd_norm_const.shape[-1]),
        variance=chi_dd_norm_const ** 2,
    )
    output_normalization_layer = keras.layers.Normalization(**normalization_kwargs)
    output_denormalization_layer = keras.layers.Normalization(
        **normalization_kwargs,
        invert=True,
    )
    y_norm = output_normalization_layer(y)
    y_norm_train = output_normalization_layer(y_train)
    y_norm_val = output_normalization_layer(y_val)
    # print("After normalization: y_norm min:\n", y_norm.min(axis=0), "\ny_norm max:\n", y_norm.max(axis=0))

    # plot the output data in three subplots
    y_norm_min, y_norm_max = y_norm.min(), y_norm.max()
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for i in range(y.shape[1]):
        axes[0].plot(ts, y_norm[:, i], label=fr"Marker {i + 1}")
        axes[1].plot(ts[:num_train_samples], y_norm_train[:, i], label=fr"Marker {i + 1}")
        axes[2].plot(ts[num_train_samples:], y_norm_val[:, i], label=fr"Marker {i + 1}")
    for i in range(3):
        axes[i].grid(True)
        axes[i].legend()
        axes[i].set_ylim(y_norm_min, y_norm_max)
    axes[0].set_title("All data")
    axes[1].set_title("Training data")
    axes[2].set_title("Validation data")
    plt.show()

    input_dim = 2 * n_chi + n_tau
    match model_type:
        case "node":
            input_dim = x.shape[-1]
            output_dim = y.shape[-1]

            layers = [
                keras.layers.Input(shape=(input_dim, )),
                input_normalization_layer,
            ]
            for _ in range(mlp_num_layers - 1):
                layers.append(keras.layers.Dense(mlp_hidden_dim, activation="tanh"))
            layers.append(keras.layers.Dense(output_dim))

            model = keras.Sequential(layers)
        case "con":
            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                input_normalization_layer,
                ConDynamics(
                    state_dim=2 * n_chi, input_dim=n_tau, input_encoding_num_layers=5, input_encoding_hidden_dim=16,
                ),
            ])
        case "lnn":
            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                input_normalization_layer,
                LnnDynamics(
                    state_dim=2 * n_chi, input_dim=n_tau, num_layers=mlp_num_layers, hidden_dim=mlp_hidden_dim,
                ),
            ])
        case _:
            raise ValueError("Invalid model type")
    model.summary()

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.AdamW(learning_rate=lr),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=norm_model_path, monitor="loss", save_best_only=True),
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    ]

    model.fit(
        x,
        y_norm,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=val_ratio,
        # validation_data=(x_val, y_norm_val),
        callbacks=callbacks,
    )

    # load the best model
    model = keras.models.load_model(norm_model_path)
    # build the model
    model.build(input_shape=(None,) + x.shape[1:])

    score_train = model.evaluate(x_train, y_norm_train, verbose=1)
    print("Training loss:", score_train)
    score_val = model.evaluate(x_val, y_norm_val, verbose=1)
    print("Validation loss:", score_val)
    # score_tot = model.evaluate(x, y_norm, verbose=1)
    # print("Score on the entire dataset:", score_tot)

    # add denormalization layer to the model
    model_with_denormalization = keras.Sequential(
        [
            model,
            output_denormalization_layer,
        ]
    )
    # print("sample output before denormalization:", model.predict(x[:1]))
    # print("sample output after denormalization:", model_with_denormalization.predict(x[:1]))

    # save the keras model
    print(f"Saving the model to {model_path.resolve()}")
    model_with_denormalization.save(model_path)
