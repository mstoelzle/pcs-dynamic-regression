import numpy as onp
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
from jax import Array, debug, random, vmap
import jax.numpy as jnp
import keras
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

from baseline_dynamical_models import ConDynamics


# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(0)


# dataset parameters
dataset_dir = Path("Source") / "Soft Robot" / "ns-2_dof-3" / "training" / "cv"
val_ratio = 0.2
num_sequences = 1000
seq_dur = 0.3
# model parameters
model_type = "node"
assert model_type in ["node", "con", "lnn"]
mlp_num_layers = 6
mlp_hidden_dim = 256
# training parameters
lr = 5e-3
batch_size = 128
num_epochs = 2500

# directory to save the model
model_dir = dataset_dir.parent.parent / "model"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / f"learned_{model_type}_model_keras_train.keras"
dynamics_model_path = model_dir / f"learned_{model_type}_model.keras"

# random seed
rng = random.PRNGKey(0)


@keras.saving.register_keras_serializable()
class OdeRollout(keras.Model):
    def __init__(self, dynamics_model: keras.Model, state_dim: int, input_dim: int, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dynamics_model = dynamics_model
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dynamics_model": keras.saving.serialize_keras_object(self.dynamics_model),
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "dt": self.dt,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        dynamics_model_config = config.pop("dynamics_model")
        dynamics_model = keras.saving.deserialize_keras_object(dynamics_model_config)
        return cls(dynamics_model, **config)

    def call(self, inputs):
        y_gt_seqs, tau_seqs = inputs[..., :self.state_dim], inputs[..., self.state_dim:]

        y_pred = y_gt_seqs[..., 0, :]
        y_pred_seqs = [y_pred]
        for time_idx in range(1, y_gt_seqs.shape[-2]):
            x = y_pred[..., :self.state_dim//2]
            x_d = y_pred[..., self.state_dim//2:self.state_dim]
            tau = tau_seqs[..., time_idx, :]

            dynamics_model_inputs = jnp.concatenate([y_pred, tau], axis=-1)
            x_dd_pred = self.dynamics_model(dynamics_model_inputs)

            # state the ODE
            y_d_pred = jnp.concatenate([x_d, x_dd_pred], axis=-1)

            # integrate the ODE with Euler's method
            y_pred = y_pred + self.dt * y_d_pred

            # append the state to the list
            y_pred_seqs.append(y_pred)
        
        y_pred_seqs = jnp.stack(y_pred_seqs, axis=-2)

        # compute the acceleration on all time steps
        y_gt_ts = y_gt_seqs.reshape(-1, self.state_dim)
        tau_ts = tau_seqs.reshape(-1, self.input_dim)
        x_dd_pred_ts = self.dynamics_model(jnp.concat([y_gt_ts, tau_ts], axis=-1))
        # reshape to batch_dim x seq_len x state_dim//2
        x_dd_pred_seqs = x_dd_pred_ts.reshape(y_gt_seqs.shape[:-1] + (self.state_dim//2,))

        output = jnp.concatenate([y_pred_seqs, x_dd_pred_seqs], axis=-1)

        return output



if __name__ == "__main__":
    # Load the data
    X = jnp.load(dataset_dir / "X.npy")  # configuration-space data
    Chi_raw = jnp.load(dataset_dir / "Chi_raw.npy")  # the raw poses in Cartesian space
    Y, Y_d = jnp.load(dataset_dir / "Y.npy"), jnp.load(dataset_dir / "Ydot.npy")
    Tau = jnp.load(dataset_dir / "Tau.npy")
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
    ts = dt * jnp.arange(num_samples)

    # marker sub-sampling
    # print("Number of markers:", num_markers)
    # marker_indices = jnp.array([num_markers - 1])
    # marker_indices = jnp.array([num_markers // 2, num_markers - 1])
    marker_indices = jnp.array([10, 15, 20])
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

    # create sequences of 0.1s duration
    seq_len = int(seq_dur / dt)
    print("Number of sequences:", num_sequences)
    ts_seqs, chi_seqs, chi_d_seqs, chi_dd_seqs, tau_seqs = [], [], [], [], []
    for seq_idx in range(num_sequences):
        # split random key
        rng, subkey1, subkey2 = random.split(rng, 3)

        # randomly select a video
        video_idx = random.randint(subkey1, (1,), 0, num_videos).item()
        # randomly select a starting frame
        video_start_frame = random.randint(subkey2, (1,), 0, num_samples_per_video - seq_len).item()

        # extract the sequence
        seq_start = video_idx * num_samples_per_video + video_start_frame
        seq_end = seq_start + seq_len

        # print("Sequence:", seq_idx, "Video:", video_idx, "Start frame:", video_start_frame, "Start:", seq_start, "End:", seq_end)

        seq_ts = ts[seq_start:seq_end]
        seq_chi_ts = Chi[seq_start:seq_end]
        seq_chi_d_ts = Chi_d[seq_start:seq_end]
        seq_chi_dd_ts = Chi_dd[seq_start:seq_end]
        seq_tau_ts = Tau[seq_start:seq_end]
        ts_seqs.append(seq_ts)
        chi_seqs.append(seq_chi_ts)
        chi_d_seqs.append(seq_chi_d_ts)
        chi_dd_seqs.append(seq_chi_dd_ts)
        tau_seqs.append(seq_tau_ts)

    # stack the sequences
    ts_seqs = jnp.stack(ts_seqs, axis=0)
    chi_seqs = jnp.stack(chi_seqs, axis=0)
    chi_d_seqs = jnp.stack(chi_d_seqs, axis=0)
    chi_dd_seqs = jnp.stack(chi_dd_seqs, axis=0)
    tau_seqs = jnp.stack(tau_seqs, axis=0)

    # Split the data into training and validation sets
    num_val_samples = int(num_sequences * val_ratio)
    num_train_samples = num_sequences - num_val_samples
    print("Number of training samples:", num_train_samples, "Number of validation samples:", num_val_samples)
    chi_seqs_train, chi_seqs_val = chi_seqs[:num_train_samples], chi_seqs[num_train_samples:]
    chi_d_seqs_train, chi_d_seqs_val = chi_d_seqs[:num_train_samples], chi_d_seqs[num_train_samples:]
    chi_dd_seqs_train, chi_dd_seqs_val = chi_dd_seqs[:num_train_samples], chi_dd_seqs[num_train_samples:]
    tau_seqs_train, tau_seqs_val = tau_seqs[:num_train_samples], tau_seqs[num_train_samples:]

    # construct the input and output data
    x = jnp.concat((chi_seqs, chi_d_seqs, tau_seqs), axis=-1)
    y = jnp.concat((chi_seqs, chi_d_seqs, chi_dd_seqs), axis=-1)
    x_train = jnp.concat((chi_seqs_train, chi_d_seqs_train, tau_seqs_train), axis=-1)
    y_train = jnp.concat((chi_seqs_train, chi_d_seqs_train, chi_dd_seqs_train), axis=-1)
    x_val = jnp.concat((chi_seqs_val, chi_d_seqs_val, tau_seqs_val), axis=-1)
    y_val = jnp.concat((chi_seqs_val, chi_d_seqs_val, chi_dd_seqs_val), axis=-1)

    # normalize the input data
    input_normalization_layer = keras.layers.Normalization(axis=-1)
    dynamics_model_input_for_normalization = jnp.concatenate([Chi, Chi_d, Tau], axis=-1)
    input_normalization_layer.adapt(dynamics_model_input_for_normalization)

    input_dim = 2 * n_chi + n_tau
    match model_type:
        case "node":
            output_dim = n_chi

            layers = [
                keras.layers.Input(shape=(input_dim, )),
                input_normalization_layer,
            ]
            for _ in range(mlp_num_layers - 1):
                layers.append(keras.layers.Dense(mlp_hidden_dim, activation="tanh"))
            layers.append(keras.layers.Dense(output_dim))

            dynamics_model = keras.Sequential(layers)
        case "con":
            dynamics_model =keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                input_normalization_layer,
                ConDynamics(state_dim=2 * n_chi, input_dim=n_tau)
            ])
        case _:
            raise ValueError("Invalid model type")
    dynamics_model.summary()

    # test the dynamics model
    dynamics_model_sample_input = jnp.concatenate([
        chi_seqs[0, 0:2], chi_d_seqs[0, 0:2], tau_seqs[0, 0:2],
    ], axis=-1)
    # print("Sample input shape:", dynamics_model_sample_input.shape)
    dynamics_model_sample_output = dynamics_model(dynamics_model_sample_input)
    # print("Sample output shape:", dynamics_model_sample_output.shape)

    model = OdeRollout(dynamics_model, state_dim=2*n_chi, input_dim=n_tau, dt=dt)
    model.summary()

    # try processing a single sequence
    sample_input = x[0:2]
    sample_target = y[0:2]
    sample_output = model(sample_input)
    print("Sample output shape:", sample_output.shape, "Sample target shape:", sample_target.shape)
    # print("sample output:\n", sample_output[0])
    # print("sample target:\n", sample_target[0])

    # loss normalization
    chi_i_norm_const = jnp.array([0.15, 0.15, 5])
    chi_norm_const = jnp.tile(chi_i_norm_const, num_markers)

    @keras.saving.register_keras_serializable()
    class NormalizedMeanSquaredError(keras.losses.Loss):
        def call(self, y_true, y_pred):
            error = y_true - y_pred
            error_x = error[..., :n_chi]
            error_x_d = error[..., n_chi:2*n_chi]
            error_x_dd = error[..., 2*n_chi:]

            # normalize the error
            norm_error_x = error_x / chi_norm_const
            norm_error_x_d = error_x_d / chi_norm_const * (0.1 * dt)
            norm_error_x_dd = error_x_dd / chi_norm_const * (0.1 * dt)**2
            
            norm_error = jnp.concatenate([norm_error_x, norm_error_x_d, norm_error_x_dd], axis=-1)
            loss = jnp.mean(jnp.square(norm_error), axis=-1)
            return loss
        

    class NormalizedRootMeanSquaredError(NormalizedMeanSquaredError):
        def call(self, y_true, y_pred):
            return jnp.sqrt(super().call(y_true, y_pred))


    # lr_schedule = keras.optimizers.schedules.CosineDecay(lr, num_epochs)
    model.compile(
        loss=[NormalizedMeanSquaredError()],
        metrics=[keras.metrics.RootMeanSquaredError()],
        optimizer=keras.optimizers.AdamW(
            # learning_rate=lr_schedule
            learning_rate=lr
        ),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True),
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    ]

    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=val_ratio,
        # validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    # load the best model
    model = keras.models.load_model(model_path)
    # build the model
    model.build(input_shape=(None, ) + x.shape[1:])
    # extract the dynamics model
    dynamics_model = model.dynamics_model

    score_train = model.evaluate(x_train, y_train, verbose=1)
    print("Training loss:", score_train)
    score_val = model.evaluate(x_val, y_val, verbose=1)
    print("Validation loss:", score_val)
    score_tot = model.evaluate(x, y, verbose=1)
    print("Score on the entire dataset:", score_tot)

    # save the keras model
    print(f"Saving the model to {model_path.resolve()}")
    dynamics_model.save(dynamics_model_path)
