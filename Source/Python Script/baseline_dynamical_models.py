from jax import Array, debug, random, vmap
import jax.numpy as jnp
import keras
from keras import ops


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
    u = ops.concatenate([a, a[n:][::-1]])
    U = u.reshape((n, n))

    # Set the elements below the diagonal to zero
    U = ops.triu(U, k=0)

    # make sure that the diagonal entries are positive
    u_diag = ops.diag(U)
    # apply shift, softplus, and epsilon
    new_u_diag = keras.activations.softplus(u_diag + diag_shift) + diag_eps
    # update diagonal
    U = U - ops.diag(u_diag) + ops.diag(new_u_diag)

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
        use_state_encoder: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.network_dim = state_dim // 2
        self.input_dim = input_dim
        self.input_encoding_num_layers = input_encoding_num_layers
        self.input_encoding_hidden_dim = input_encoding_hidden_dim
        self.diag_shift, self.diag_eps = 1e-6, 2e-6

        if use_state_encoder:
            # linear encoder
            self.state_encoder = keras.layers.Dense(self.network_dim, kernel_initializer="identity", bias_initializer="zeros")
        else:
            self.state_encoder = None

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
                V_layers = [keras.layers.Input(shape=(self.input_dim, ))]
                for _ in range(input_encoding_num_layers - 1):
                    V_layers.append(keras.layers.Dense(input_encoding_hidden_dim, activation="tanh"))
                V_layers.append(keras.layers.Dense(self.network_dim * self.input_dim))
                self.V_nn = keras.Sequential(V_layers)
            elif self.network_dim== self.input_dim:
                self.V_nn = lambda tau: ops.eye(self.network_dim)[None, ...].repeat(tau.shape[0], axis=0)
            else:
                self.V_nn = lambda tau: ops.zeros((tau.shape[0], self.network_dim, self.input_dim))
        else:
            self.V_nn = lambda tau: ops.zeros_like(tau)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "input_encoding_num_layers": self.input_encoding_num_layers,
                "input_encoding_hidden_dim": self.input_encoding_hidden_dim,
                "use_state_encoder": self.state_encoder is not None,
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

        if self.state_encoder is not None:
            z = self.state_encoder(x)
            z_d = ops.matmul(x_d, self.state_encoder.kernel)
        else:
            z, z_d = x, x_d

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
        # eig = ops.linalg.eigh(Gamma_w)[0]
        # print(f"Eigenvalues: {eig}")
        # debug.print("Eigenvalues: {eig}", eig=eig)

        # print("u", u[0])
        # print("Gamma w times x", Gamma_w[0] @ x[0])
        # print("E w times x_d", E_w[0] @ x_d[0])
        # print("tanh x", ops.tanh(x[0] + self.bias))

        # compute the acceleration of the oscillator network
        # x_dd = vmap(
        #     lambda _x, _x_d, _u: B_w_inv @ (
        #         _u
        #         - Gamma_w @ _x
        #         - E_w @ _x_d
        #         -ops.tanh(_x + self.bias)
        #     )
        # )(x, x_d, u)
        z_dd = ops.einsum("bij,bj->bi",
            B_w_inv,
            u
            - ops.einsum("bij,bj->bi", Gamma_w, z)
            - ops.einsum("bij,bj->bi", E_w, z_d)
            - ops.tanh(z + self.bias)
        )

        if self.state_encoder is not None:
            x_dd = ops.matmul(z_dd, ops.linalg.inv(self.state_encoder.kernel))
        else:
            x_dd = z_dd

        # print("z_dd mean", z_dd.mean(), "x_dd", x_dd.mean())

        return x_dd

    def input_state_coupling(self, tau: Array) -> Array:
        V = self.V_nn(tau).reshape(-1, self.network_dim, self.input_dim)
        return V

    def encode_input(self, tau: Array):
        V = self.input_state_coupling(tau)
        u = ops.einsum("bij,bj->bi", V, tau)
        # u = V @ tau[: self.input_dim]
        # u = vmap(lambda _V, _tau: _V @ _tau)(V, tau)

        return u