import numpy as np
from .func_fit import FuncFit
import itertools
#import traceback

class XOR(FuncFit):
    def __init__(self, num_inputs, bias_value=None, bias_func=None, error_method="mse"):
        super().__init__(error_method=error_method)
        self.num_inputs = num_inputs
        self.bias_value = bias_value  # Optional fixed bias value
        self.bias_func = bias_func    # Optional bias function
        self.input_vector = self.generate_xor_inputs(self.num_inputs, self.bias_value, self.bias_func) # Generate inputs with bias based on the provided bias_value and bias_func

        # Compute XOR dynamically for all inputs (ignoring the bias column if present)
        xor_outputs = []
        for inp in self.inputs:
            # If bias_value or bias_func is not None, ignore the last column (bias) for XOR calculation
            input_vector = inp[:-1] if self.bias_value is not None or self.bias_func is not None else inp
            xor_result = self.compute_xor(input_vector)
            xor_outputs.append([xor_result])
        self.output_vector = np.array(xor_outputs, dtype=np.float32)

    @property
    def inputs(self):
        return self.input_vector

    @property
    def targets(self):
        return self.output_vector

    def generate_xor_inputs(self, num_inputs, bias_value=None, bias_func=None):
        """Generate all possible binary input vectors of length `num_inputs`. Optionally append a bias term."""
        binary_combinations = list(itertools.product([0, 1], repeat=num_inputs))
        inputs = np.array(binary_combinations, dtype=np.float32)
        
        if bias_value is None and bias_func is None:
            return inputs
        if bias_value is not None and bias_func is not None:
            print("Warning: Both bias_value and bias_func are set. Using bias_value and ignoring bias_func.")
        if bias_value is not None:
            bias = np.full((inputs.shape[0], 1), bias_value, dtype=np.float32)
        elif bias_func is not None:
            bias = np.array([[bias_func()] for _ in range(inputs.shape[0])], dtype=np.float32)
        
        return np.hstack([inputs, bias])

    def compute_xor(self, input_vector):
        """Compute XOR dynamically for any number of binary inputs."""
        result = int(input_vector[0])  # Start with the first element
        for value in input_vector[1:]:
            result ^= int(value)  # Apply XOR with the next element
        return float(result)

    @property
    def input_shape(self):
        # If bias is used, include it in the input shape
        input_length = self.num_inputs + (1 if self.bias_value is not None or self.bias_func is not None else 0)
        return (2 ** self.num_inputs, self.num_inputs)  # Match input shape for XOR

    @property
    def output_shape(self):
        return (2 ** self.num_inputs, 1)
