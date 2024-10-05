import jax
from jax import vmap, numpy as jnp
import jax.numpy as jnp

from ..base import BaseProblem
from tensorneat.common import State


class FuncFit(BaseProblem):
    jitable = True

    def __init__(self, error_method: str = "mse"):
        super().__init__()

        assert error_method in {"mse", "rmse", "mae", "mape", "pureples"}
        self.error_method = error_method

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):

        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )

        # Using match-case for handling error method
        match self.error_method:
            case "mse":
                loss = jnp.mean((predict - self.targets) ** 2)

            case "rmse":
                loss = jnp.sqrt(jnp.mean((predict - self.targets) ** 2))

            case "mae":
                loss = jnp.mean(jnp.abs(predict - self.targets))

            case "mape":
                loss = jnp.mean(jnp.abs((predict - self.targets) / self.targets))

            case "pureples":
                # Pureples logic: compute sum square error for XOR problem
                sum_square_error = jnp.sum((predict - self.targets) ** 2) / float(len(self.targets))
                # Fitness in Pureples is calculated as: 1 - sum_square_error
                loss = 1.0 - sum_square_error

            case _:
                raise NotImplementedError(f"Error method {self.error_method} not implemented")

        # Return -loss for minimization (in both Pureples and TensorNEAT)
        return -loss


    """ def evaluate(self, state, randkey, act_func, params):

        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )

        if self.error_method == "mse":
            loss = jnp.mean((predict - self.targets) ** 2)

        elif self.error_method == "rmse":
            loss = jnp.sqrt(jnp.mean((predict - self.targets) ** 2))

        elif self.error_method == "mae":
            loss = jnp.mean(jnp.abs(predict - self.targets))

        elif self.error_method == "mape":
            loss = jnp.mean(jnp.abs((predict - self.targets) / self.targets))

        else:
            raise NotImplementedError

        return -loss """

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        fitness = self.evaluate(state, randkey, act_func, params)

        loss = -fitness

        msg = ""
        for i in range(inputs.shape[0]):
            msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError
