import random
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union, Optional
import numpy as np
from ConfigSpace.api.types.float import Float
from ConfigSpace.api.types.integer import Integer
from ConfigSpace.api.types.categorical import Categorical


class AbstractBase(ABC):
    def __init__(self, parameter: Optional[Union[Float, Integer, Categorical]], generator: np.random.Generator) -> None:
        """
        Initializes the function with a specific parameter configuration and generator.
        """
        self.parameter = parameter
        self.generator = generator

    @property
    def x_min(self) -> Optional[np.ndarray]:
        pass

    @property
    def f_min(self) -> Any:
        pass

    @abstractmethod
    def evaluate(self, x) -> Any:
        """
        Abstract method for evaluating the function at a given point.
        """
        pass


class NoisyFunction(AbstractBase):
    """
    Represents a function that generates random noise within a specified range.
    This function does not utilize the `parameter` attribute for its noise generation.
    """

    def __init__(self, parameter: Optional[Union[Float, Integer, Categorical]], generator: np.random.Generator) -> None:
        """
        Initializes the NoisyFunction with a given seed.

        Args:
            seed (int): Seed for random number generation.
        """
        self.seed = generator.integers(1000, size=1)[0].item()
        super().__init__(parameter=parameter, generator=generator)

    def evaluate(self, x: Optional[Any] = None) -> float:
        """
        Evaluates the noisy function, generating a random value within the range [-1, 1].

        Args:
            x (Optional[Any], optional): Ignored for this function. Defaults to None.

        Returns:
            float: A random float within the range [-1, 1].
        """
        random.seed(self.seed + (round(x) if x is not None else 0))
        return random.uniform(-1, 1)

    @property
    def x_min(self) -> np.ndarray | None:
        return None

    @property
    def f_min(self) -> Any:
        return -1


class QuadraticFunction(AbstractBase):
    """
    Represents a quadratic function defined by coefficients a, b, and c.
    """

    def __init__(self, parameter: Optional[Union[Float, Integer, Categorical]], generator: np.random.Generator) -> None:
        """
        Initializes a QuadraticFunction instance with randomly generated coefficients.
        """
        self.a, self.b, self.c = generator.integers(-1000, 1000, 3)
        super().__init__(parameter, generator)


    def get_parameters(self) -> Tuple[int, int, int]:
        """
        Returns the coefficients of the quadratic function.

        Returns:
            Tuple[int, int, int]: The coefficients a, b, and c of the function.
        """
        return self.a, self.b, self.c

    def evaluate(self, x: float) -> float:
        """
        Evaluates the quadratic function at a given value of x.
        """
        return self.a * x**2 + self.b * x + self.c

    @property
    def x_min(self):
        """
        Calculates the vertex of the quadratic function as its optimum, considering the function's domain.
        """
        # Domain boundaries
        lower_x = self.parameter.lower
        upper_x = self.parameter.upper

        lower_f = self.evaluate(lower_x)
        upper_f = self.evaluate(upper_x)

        if self.a == 0:
            return lower_x if lower_f < upper_f else upper_x

        # Calculate the x value where the minimum would occur if it were in the bounds
        x_vertex = - self.b / (2 * self.a)

        # If a > 0, the vertex is a minimum; check if it’s within bounds
        if self.a > 0 and lower_x <= x_vertex <= upper_x:
            return x_vertex
        else:
            return lower_x if lower_f < upper_f else upper_x

    @property
    def f_min(self, x_min: Optional[float] = None):
        if not x_min:
            x_min = self.x_min
        return self.evaluate(x_min)
