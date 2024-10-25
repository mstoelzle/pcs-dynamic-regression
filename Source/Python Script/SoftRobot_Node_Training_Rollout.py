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
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dynamics_model = dynamics_model
        self.dt = dt

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "dynamics_model": keras.saving.serialize_keras_object(self.dynamics_model),
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

def generate_positive_definite_matrix_from_params(
    n: int, a: Array, diag_shift: float = 1e-6, diag_eps: float = 2e-6
) -> Array:
    """
    Generate a positive definite matrix of shape (n, n) from a vector of parameters.
    Args:
        n: Number of rows and columns of the matrix.
        a: A vector of parameters of shape ((n^2 + n) / 2, ).
        diag_shift: A small value that is added to the diagonal entries of the matrix before the softplus.
        diag_eps: A small value that is added to the diagonal entries of the matrix after the softplus.
    Returns:
        A: A positive definite matrix of shape (n, n).
    """
    # construct upper triangular matrix
    # https://github.com/google/jax/discussions/10146
    u = jnp.concatenate([a, a[n:][::-1]])
    U = u.reshape((n, n))

    # Set the elements below the diagonal to zero
    U = jnp.triu(U, k=0)

    # make sure that the diagonal entries are positive
    u_diag = jnp.diag(U)
    # apply shift, softplus, and epsilon
    new_u_diag = keras.activations.softplus(u_diag + diag_shift) + diag_eps
    # update diagonal
    U = U - jnp.diag(u_diag) + jnp.diag(new_u_diag)

    # reverse Cholesky decomposition
    A = U.transpose() @ U

    return A

@keras.saving.register_keras_serializable()
class ConDynamics(keras.Model):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        input_encoding_num_layers: int = 5,
        input_encoding_hidden_dim: int = 32,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.network_dim = state_dim // 2
        self.input_dim = input_dim
        self.input_encoding_num_layers = input_encoding_num_layers
        self.input_encoding_hidden_dim = input_encoding_hidden_dim
        self.diag_shift, self.diag_eps = 1e-6, 2e-6

        # number of params in B_w / B_w_inv matrix
        num_b_w_params = int((self.network_dim ** 2 + self.network_dim) / 2)
        # constructing B_w_inv as a positive definite matrix
        self.b_w_inv = self.add_weight(
            shape=(num_b_w_params,),
            initializer="glorot_normal",
            trainable=True,
            name="b_w_inv",
        )

        # constructing Lambda_w as a positive definite matrix
        num_gamma_w_params = int((self.network_dim ** 2 + self.network_dim) / 2)
        # vector of parameters for triangular matrix
        self.gamma_w = self.add_weight(
            shape=(num_gamma_w_params, ),
            initializer="glorot_normal",
            trainable=True,
            name="gamma_w",
        )

        # constructing E_w as a positive definite matrix
        num_e_w_params = int((self.network_dim ** 2 + self.network_dim) / 2)
        # vector of parameters for triangular matrix
        self.e_w = self.add_weight(
            shape=(num_e_w_params, ),
            initializer="glorot_normal",
            trainable=True,
            name="gamma_w",
        )

        # bias term
        self.bias = self.add_weight(
            shape=(self.network_dim, ),
            trainable=True,
            name="bias",
        )

        if self.input_dim > 0:
            if input_encoding_num_layers > 0:
                V_layers = [keras.layers.InputLayer(input_shape=(self.input_dim, ))]
                for _ in range(input_encoding_num_layers - 1):
                    V_layers.append(keras.layers.Dense(input_encoding_hidden_dim, activation="tanh"))
                V_layers.append(keras.layers.Dense(self.network_dim * self.input_dim))
                self.V_nn = keras.Sequential(V_layers)
            elif self.network_dim== self.input_dim:
                self.V_nn = lambda tau: jnp.eye(self.network_dim)[None, ...].repeat(tau.shape[0], axis=0)
            else:
                self.V_nn = lambda tau: jnp.zeros((tau.shape[0], self.network_dim, self.input_dim))
        else:
            self.V_nn = lambda tau: jnp.zeros_like(tau)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "input_encoding_num_layers": self.input_encoding_num_layers,
                "input_encoding_hidden_dim": self.input_encoding_hidden_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        batch_size = inputs.shape[0]
        y, tau = inputs[..., :self.state_dim], inputs[..., self.state_dim:]
        x, x_d = y[..., :self.network_dim], y[..., self.network_dim:]

        # compute the oscillator network input
        u = self.encode_input(tau)

        # compute the PD matrices
        Gamma_w = generate_positive_definite_matrix_from_params(
            self.network_dim,
            self.gamma_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )[None, ...].repeat(batch_size, axis=0)
        E_w = generate_positive_definite_matrix_from_params(
            self.network_dim,
            self.e_w,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )[None, ...].repeat(batch_size, axis=0)
        B_w_inv = generate_positive_definite_matrix_from_params(
            self.network_dim,
            self.b_w_inv,
            diag_shift=self.diag_shift,
            diag_eps=self.diag_eps,
        )[None, ...].repeat(batch_size, axis=0)

        # eigenvalues of the matrix
        # eig = jnp.linalg.eigh(Gamma_w)[0]
        # print(f"Eigenvalues: {eig}")
        # debug.print("Eigenvalues: {eig}", eig=eig)

        # print("u", u[0])
        # print("Gamma w times x", Gamma_w[0] @ x[0])
        # print("E w times x_d", E_w[0] @ x_d[0])
        # print("tanh x", jnp.tanh(x[0] + self.bias))

        # compute the acceleration of the oscillator network
        # x_dd = vmap(
        #     lambda _x, _x_d, _u: B_w_inv @ (
        #         _u
        #         - Gamma_w @ _x
        #         - E_w @ _x_d
        #         -jnp.tanh(_x + self.bias)
        #     )
        # )(x, x_d, u)
        x_dd = jnp.einsum("bij,bj->bi",
            B_w_inv,
            u
            - jnp.einsum("bij,bj->bi", Gamma_w, x)
            - jnp.einsum("bij,bj->bi", E_w, x_d)
            - jnp.tanh(x + self.bias)
        )

        return x_dd

    def input_state_coupling(self, tau: Array) -> Array:
        V = self.V_nn(tau).reshape(-1, self.network_dim, self.input_dim)
        return V

    def encode_input(self, tau: Array):
        V = self.input_state_coupling(tau)
        u = jnp.einsum("bij,bj->bi", V, tau)
        # u = V @ tau[: self.input_dim]
        # u = vmap(lambda _V, _tau: _V @ _tau)(V, tau)

        return u



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
