import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 8,
        use_bias: bool = False,  # set False to revert to original (no bias)
    ):
        """
        A minimalistic Neural Network, using numpy.
        - One hidden layer with tanh activation
        - Output layer with tanh activation

        :param int input_size: Size of input vector
        :param int hidden_size: Size of hidden layer
        :param int output_size: Size of output vector
        :param bool use_bias: Whether to include bias terms
        """
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size
        self.use_bias = use_bias
        self.n_con1 = input_size * hidden_size
        self.n_con2 = hidden_size * output_size
        self.lin = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias1 = np.zeros(hidden_size) if use_bias else None
        self.output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.bias2 = np.zeros(output_size) if use_bias else None
        self.n_params = self.get_num_params()

    def get_action(self, state):
        assert state.shape[-1] == self.n_input, (
            "State does not correspond with expected input size"
        )

        hid_l = state @ self.lin.T
        if self.use_bias:
            hid_l = hid_l + self.bias1
        hid_l = np.tanh(hid_l)

        output_l = hid_l @ self.output.T
        if self.use_bias:
            output_l = output_l + self.bias2
        output_l = np.tanh(output_l)

        return np.clip(output_l, -1.0, 1.0)

    def set_weights(self, weights):
        """
        Set weights of NN.

        :param np.ndarray weights: Vector of weights
        """
        assert len(weights) == self.n_params, (
            f"Got {len(weights)} but expected {self.n_params}"
        )
        idx = 0
        self.lin = weights[idx:idx + self.n_con1].reshape(self.lin.shape)
        idx += self.n_con1
        if self.use_bias:
            self.bias1 = weights[idx:idx + self.n_hidden]
            idx += self.n_hidden
        self.output = weights[idx:idx + self.n_con2].reshape(self.output.shape)
        idx += self.n_con2
        if self.use_bias:
            self.bias2 = weights[idx:idx + self.n_output]

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        """Return the total number of parameters in the network."""
        n = self.n_con1 + self.n_con2
        if self.use_bias:
            n += self.n_hidden + self.n_output
        return n

    def reset_controller(self, batch_size=1) -> None:
        pass
