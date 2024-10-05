import numpy as np

from .func_fit import FuncFit


class XOR2d_bias(FuncFit):
    @property
    def inputs(self):
        original_inputs = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            dtype=np.float32,
        )
        # Automatically append bias term (1) to each input vector
        bias = np.ones((original_inputs.shape[0], 1), dtype=np.float32)
        return np.hstack([original_inputs, bias])

    @property
    def targets(self):
        return np.array(
            [[0], [1], [1], [0]],
            dtype=np.float32,
        )

    @property
    def input_shape(self):
        return 4, 3

    @property
    def output_shape(self):
        return 4, 1
