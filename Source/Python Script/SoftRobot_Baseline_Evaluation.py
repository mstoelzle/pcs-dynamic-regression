import jax
import numpy as np
import os

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
os.environ["KERAS_BACKEND"] = "jax"

from jax import Array, jit, vmap
import jax.numpy as jnp
# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter


# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(0)

# define directories
case_dir = Path(f"./Source/Soft Robot/ns-2_dof-3")
dataset_dir = case_dir / "training" / "cv"
model_dir = case_dir / "model"
print("Model dir", model_dir.resolve())

# load the trained model
learned_model = keras.models.load_model(str(model_dir / 'learned_node_model.keras'), safe_mode=False)


def ode_fn(t: Array, y: Array, tau: Array) -> Array:
    chi, chi_d = jnp.split(y, 2)
    model_input = jnp.concatenate([chi, chi_d, tau], axis=-1)[None, :]

    chi_dd = learned_model.predict(model_input, batch_size=1).squeeze(axis=0)

    y_d = jnp.concatenate([chi_d, chi_dd], axis=-1)
    return y_d


if __name__ == "__main__":
    # load the dataset
    Y = jnp.load(dataset_dir / 'Y.npy')
    Y_d = jnp.load(dataset_dir / 'Ydot.npy')
    Tau = np.load(dataset_dir / "Tau.npy")
    Tau = Tau.reshape(Y.shape[0], Tau.shape[-1])

    # set the time steps
    dt = 1e-3
    ts = dt * jnp.arange(Y.shape[0])

    # split the dataset
    n_chi = Y.shape[-1] // 2
    Chi, Chi_d = Y[:, :n_chi], Y[:, n_chi:]
    Chi_dd = Y_d[:, n_chi:]

    # define the initial condition
    y0 = jnp.array(Y[0])
    tau0 = jnp.array(Tau[0])

    # test the ode function
    y_d0 = ode_fn(0.0, y0, tau0)
    print(y_d0)
