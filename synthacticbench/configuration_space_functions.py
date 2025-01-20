from __future__ import annotations

import math

import numpy as np
from ConfigSpace import (
    Categorical,
    CategoricalHyperparameter,
    ConfigurationSpace,
    Float,
    Integer,
)
from numpy import ndarray

from synthacticbench.abstract_function import AbstractFunction
from synthacticbench.base_functions import (
    Ackley,
    Griewank,
    Rosenbrock,
    SumOfQ,
)


class RelevantParameters(AbstractFunction):
    """
    Configuration Space Benchmark - C1
    This benchmark models a function where only a subset of parameters significantly affects
    the optimization task.

    This class calculates a sum of quadratic functions and noisy parameters
    to simulate a scenario where an optimizer is tasked with identifying relevant parameters
    among a large set of potential ones.
    """

    def __init__(
        self,
        num_quadratic: int,
        dim: int,
        noise_low: float = -100,
        noise_high=100,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark with the specified dimensions and noise level.

        Args:
            num_quadratic (int): Number of quadratic functions representing relevant
                parameters.
            dim (int): Total dimensionality of the parameter space.
            noise_low (float, optional): Lower bound for the noisy parameters' range.
                Default is -100.
            noise_high (float, optional): Upper bound for the noisy parameters' range.
                Default is 100.
            seed (int or None, optional): Seed for random number generation. Default is None.
            loggers (list or None, optional): Optional list of loggers for logging.
                Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.num_quadratic = num_quadratic
        self.num_noisy = self.dim - self.num_quadratic

        self.rng = np.random.default_rng(seed=seed)

        self.instance = SumOfQ(seed=seed, dim=self.num_quadratic)
        self.noisy_function = self.rng.spawn(n_children=1)[0]

        self._configspace = self._create_config_space()

        self.benchmark_name = "c1"

    def _create_config_space(self):
        quadr_space = self.instance._create_config_space()

        quadr_space.add_configuration_space(
            configuration_space=ConfigurationSpace(
                {
                    f"n_{i}": Float(
                        bounds=(-100, 100),
                        default=0,
                        name=f"n_{i}",
                    )
                    for i in range(self.num_noisy)
                },
                seed=self.seed,
            ),
            prefix="",
            delimiter="",
        )
        return quadr_space

    def _function(self, x: ndarray) -> float:
        quadr_sum = self.instance._function(x[: self.num_quadratic])
        noisy_sum = sum(
            self.noisy_function.uniform(
                low=self.noise_low, high=self.noise_high, size=self.num_noisy
            )
        )

        return quadr_sum + noisy_sum

    @property
    def x_min(self):
        """
        The minimum parameter values for the benchmark, combining the minimum values for the
        quadratic parameters and the noisy parameters. Because the noisy parameters do not
        have a meaningful minimum, their x_min is denoted by None
        (reflecting the entire domain).

        Returns:
            ndarray: An array of minimum parameter values,
                with `None` for the noisy parameters.
        """
        x_mins_quadr = self.instance.x_min
        x_mins_noisy = np.array([None] * self.num_noisy)

        return np.concatenate((x_mins_quadr, x_mins_noisy), axis=0)

    @property
    def f_min(self) -> float:
        """
        The minimum value of the objective function, which combines the minimum values of
        the quadratic function and the expected value of the noisy contribution.

        Returns:
            float: The minimum value of the objective function.
        """
        return self.instance.f_min


class ParameterInteractions(AbstractFunction):
    """
    Parameter Interactions - C2
    This benchmark resembles algorithm configuration scenarios where parameters are
    interdependent and require joint tuning to fully leverage interaction effects.
    It includes various mathematical test functions, specifically designed to
    exhibit such behavior.
    """

    def __init__(
        self,
        dim: int,
        name: str = "ackley",
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark with the specified dimension.

        Args:
            dim (int): Total dimensionality of the parameter space.
            name (str): Type of function to be used. Must be one of "ackley", "rosenbrock" or
                "griewank". Defaults to "ackley".
            seed (int or None, optional): Seed for random number generation. Default is None.
            loggers (list or None, optional): Optional list of loggers for logging.
                Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )
        self.benchmark_name = "c2"
        self.name = name
        self.dim = dim
        self.rng = np.random.default_rng(seed=seed)

        if self.name.lower() == "rosenbrock":
            self.instance = Rosenbrock(dim, seed)
        elif self.name.lower() == "ackley":
            self.instance = Ackley(dim, seed)
        elif self.name.lower() == "griewank":
            self.instance = Griewank(dim, seed)

        self._configspace = self.instance._create_config_space()

    def _function(self, x: np.ndarray) -> np.ndarray:
        return self.instance._function(x)

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self):
        return self.instance.f_min


class MixedTypes(AbstractFunction):
    """
    Configuration Space Benchmark - C3
    In this benchmark, we investigate to what extent an optimizer is capable of dealing with
    configuration spaces comprising mixed types of parameters. We distinguish between
    categorical, Boolean (as a special instance of categorical), integer, and float parameters.
    """

    def __init__(
        self,
        dim: int,
        share_cat: float,
        share_bool: float,
        share_int: float,
        share_float: float,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): Total dimensionality of the parameter space.
            share_cat (float): Share of categorical parameters.
            share_bool (float): Share of boolean parameters.
            share_int (float): Share of integer parameters.
            share_float (float): Share of float parameters.
            seed (int or None, optional): Seed for random number generation. Default is None.
            loggers (list or None, optional): Optional list of loggers for logging.
                Default is None.

        Note:
            The share_cat/bool/int/float parameters are normalized over the
            number of all parameters to give the share of that parameter type over the
            dimension dim.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )

        # normalize shares
        share_sum = share_cat + share_bool + share_int + share_float
        self.share_cat = share_cat / share_sum
        self.share_bool = share_bool / share_sum
        self.share_int = share_int / share_sum
        self.share_float = share_float / share_sum

        self.instance = SumOfQ(seed, dim, loggers=loggers)
        self.lower_bounds = [-100] * self.dim
        self.upper_bounds = [100] * self.dim
        self.groups = []

        self.rng = np.random.default_rng(seed=seed)

        self.benchmark_name = "c3"
        self._configspace = self._create_config_space()

    def _create_config_space(self):
        cs = ConfigurationSpace(seed=self.seed)
        i = 0

        # add Categorical hyperparameters
        j = 0
        for j in range(math.floor(self.share_cat * self.dim)):
            name = f"x_{i+j}"
            group = self.rng.integers(low=3, high=21)
            self.groups.append(group)
            cs.add(Categorical(name=name, items=range(group), default=0))
        i += j + 1

        # add Boolean hyperparameters
        j = 0
        for j in range(math.floor(self.share_bool * self.dim)):
            name = f"x_{i+j}"
            cs.add(Categorical(name=name, items=range(2), default=0))
        i += j + 1

        # add Integer hyperparameters
        j = 0
        for j in range(math.floor(self.share_int * self.dim)):
            name = f"x_{i+j}"
            cs.add(Integer(name=name, bounds=(-100, 100), default=0))
        i += j + 1

        # add Float hyperparameters
        for j in range(i, self.dim):
            name = f"x_{i+j}"
            cs.add(
                Float(
                    bounds=(self.lower_bounds[j], self.upper_bounds[j]),
                    default=0.5,
                    name=f"x_{i+j}",
                )
            )

        return cs

    def _function(self, x: np.ndarray) -> np.ndarray:
        i = 0
        j = 0
        # transform categorical values
        for j in range(math.floor(self.share_cat * self.dim)):
            slice_size = (self.upper_bounds[j] - self.lower_bounds[j]) / self.groups[j]
            offset = x[j] * slice_size
            x[j] = self.lower_bounds[j] + offset
        i += j + 1

        # transform Boolean values
        for j in range(math.floor(self.share_bool * self.dim)):
            slice_size = (self.upper_bounds[j] - self.lower_bounds[j]) / 4
            offset = (x[i + j] + 1) * slice_size
            x[j] = self.lower_bounds[i + j] + offset

        # query base function for transformed x
        return self.instance._function(x)


class ActivationStructures(AbstractFunction):
    """
    Activation Structures - C4
    This benchmark simulates algorithm configuration scenarios where a set of parameters can
    be grouped into distinct subsets each of which is active if and only if a categorical
    parameter takes a certain value.

    The categorical parameter is supposed to be the last entry of the input vector.
    """

    def __init__(
        self,
        dim: int,
        groups: int = 1,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): Total dimensionality of the parameter space.
            groups (int): Number of subsets of parameters that can be activated by different
                values of the a categorical parameter. Defaults to 1.
            seed (int or None, optional): Seed for random number generation. Default is None.
            loggers (list or None, optional): Optional list of loggers for logging.
                Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )

        self._x_min = None

        assert groups > 0, "Benchmark must have at least one group."
        self.groups = groups
        self.rng = np.random.default_rng(seed=seed)

        dim_without_categorical = self.dim - 1

        min_group_size = dim_without_categorical // self.groups
        remainder = dim_without_categorical % self.groups
        group_dims = [min_group_size + 1] * remainder + [min_group_size] * (
            self.groups - remainder
        )
        self.group_dims = group_dims

        self.instances = self._make_groups()

        self._configspace = self._create_config_space()
        self.benchmark_name = "c4"

    def _create_config_space(self):
        configuration_space = ConfigurationSpace()

        for cat, instance in self.instances.items():
            function = instance["function"]

            configuration_space.add_configuration_space(
                configuration_space=function._create_config_space(),
                prefix=cat,
                delimiter=":",
            )
        configuration_space.add(
            CategoricalHyperparameter(name="c", choices=range(self.groups), default_value=0)
        )
        return configuration_space

    def _make_groups(self):
        instances = {}
        x_i = 0

        group_seeds = self.rng.integers(low=0, high=1000, size=self.groups)

        for i in range(self.groups):
            instances[i] = {
                "function": SumOfQ(seed=group_seeds[i], dim=self.group_dims[i]),
                "starts_at": x_i,
            }
            x_i += self.group_dims[i]

        return instances

    def _function(self, x: np.ndarray) -> float:
        cat = x[-1]
        instance = self.instances[cat]["function"]
        starts_at = self.instances[cat]["starts_at"]
        x = x[starts_at : starts_at + instance.dim]
        return instance._function(x)

    def _calculate_x_min(self):
        x_mins = [[None] * group_dim for group_dim in self.group_dims]
        x_min_init = [item for sublist in x_mins for item in sublist]

        current_best_f = math.inf

        cat_of_x_min = None
        for cat, func in self.instances.items():
            if func["function"].f_min < current_best_f:
                cat_of_x_min = cat
                x_min = [
                    float(val) if isinstance(val, np.float64) else val
                    for val in func["function"].x_min
                ]

        instance = self.instances[cat_of_x_min]["function"]
        starts_at = self.instances[cat_of_x_min]["starts_at"]
        x_min_init[starts_at : starts_at + instance.dim] = x_min
        x_min = [*x_min_init, cat_of_x_min]
        self._x_min = x_min

    @property
    def x_min(self) -> np.ndarray | None:
        """
        Return the cached x_min or calculate it if not already done.

        Returns:
            np.ndarray | None: The global minimum location (x_min) as a numpy array.
        """
        if self._x_min is None:
            self._calculate_x_min()

        return np.array(self._x_min)

    @property
    def f_min(self) -> float:
        return self._function(self.x_min)


class ShiftingDomains(AbstractFunction):
    """
    Shifting Domains - C5
    This benchmark models parameter domain shifts, where the choice of one parameter's (x0)
    value affects the valid domains of other parameters. This class simulates an optimization
    problem in which the parameter search space can change dynamically based on the values of
    certain parameters.
    """

    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the ShiftingDomains class, setting up two instances of the SumOfQ
        function—one for the original domains and one for the shifted domains.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )
        self.benchmark_name = "c5"
        self.rng = np.random.default_rng(seed=seed)

        assert self.dim > 1, "Dimension has to be larger than one in this problem"

        lower_bounds = [-100].append([0] * (self.dim - 1))
        upper_bounds = [100].append([200] * (self.dim - 1))

        self.instance = SumOfQ(seed=seed, dim=dim)
        self.instance_shifted_domains = SumOfQ(
            seed=seed, dim=dim, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )

        self._configspace = self._create_config_space(instance=self.instance)
        self._configspace_shifted_domains = self._create_config_space(
            instance=self.instance_shifted_domains
        )

    def _create_config_space(self, instance: SumOfQ):
        """
        Creates the configuration space based on the given SumOfQ instance.

        Args:
            instance (SumOfQ): The SumOfQ instance to create the config space.

        Returns:
            ConfigurationSpace:
                A configuration space defining the parameter bounds and default values.
        """
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(
                        instance.lower_bounds[i],
                        instance.upper_bounds[i],
                    ),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _function(self, x: ndarray) -> float:
        """
        Evaluates the objective function for the given x. If the x0 is less than 0,
        the original configuration space is used.
        Otherwise, the shifted configuration space is used.

        Args:
            x (ndarray): A vector of parameter values to evaluate.

        Returns:
            float: The value of the objective function based on the current configuration
                space (original or shifted).
        """
        # Check the value of x0
        if x[0] < 0:
            return self.instance._function(x=x)

        # Domains have shifted! Use shifted domains function
        self._configspace = self._configspace_shifted_domains
        return self.instance_shifted_domains._function(x=x)

    @property
    def x_min(self) -> np.ndarray | None:
        """
        Returns the optimal parameter vector `x_min` for the benchmark function.
        This is the optimal vector from either the original or shifted domains, depending on
        which yields the smaller minimum value.

        Returns:
            np.ndarray | None: The optimal parameter vector.
        """
        if self.instance.f_min <= self.instance_shifted_domains.f_min:
            return self.instance.x_min
        return self.instance_shifted_domains.x_min

    @property
    def f_min(self) -> float:
        """
        Returns the minimum value of the objective function `f_min`. This is the smaller of
        the minimum values from the original and shifted domains.

        Returns:
            float: The minimum value of the objective function.
        """
        return min(self.instance.f_min, self.instance_shifted_domains.f_min)


class HierarchicalStructures(AbstractFunction):
    """
    Hierarchical Structures - C6
    This benchmark simulates algorithm configuration scenarios with a hierarchical structure:
    a set of parameters is grouped into distinct subsets (groups),
    each of which is further divided into subgroups,
    where a pair of categorical parameters determines the active subset.

    The categorical parameter determining the active group is supposed to be the last entry of
    the input vector and the second last determines the active subgroup within the
    active group.
    """

    def __init__(
        self,
        dim: int,
        groups: int,
        subgroups_per_group: int,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            groups (int): Number of groups.
            subgroups_per_group (int): Number of subgroups in each group.
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )
        assert groups * subgroups_per_group <= dim - 2, (
            "The total number of subgroups "
            f"(groups * subgroups_per_group = {groups * subgroups_per_group}) "
            f"must not exceed the total number of non-categorical parameters ({dim-2}). "
        )
        self._x_min = None
        self.groups = groups
        self.subgroups_per_group = subgroups_per_group
        self.rng = np.random.default_rng(seed=seed)
        # Determine dimensions for groups and subgroups
        self._init_dims()

        self.instances = self._make_hierarchical_groups()

        self._configspace = self._create_config_space()
        self.benchmark_name = "c6"

    def _init_dims(self):
        self.total_subgroups = self.groups * self.subgroups_per_group
        self.dim_without_categorical = self.dim - 2
        min_group_size = self.dim_without_categorical // self.groups
        remainder = self.dim_without_categorical % self.groups
        group_dims = [min_group_size + 1] * remainder + [min_group_size] * (
            self.groups - remainder
        )
        self.group_dims = group_dims
        self.all_subgroup_dims = []
        for group_dim in group_dims:
            min_group_size = group_dim // self.subgroups_per_group
            remainder = group_dim % self.subgroups_per_group
            subgroup_dims = [min_group_size + 1] * remainder + [min_group_size] * (
                self.subgroups_per_group - remainder
            )
            self.all_subgroup_dims.append(subgroup_dims)

    def _create_config_space(self):
        configuration_space = ConfigurationSpace()

        for group_id, group_instances in self.instances.items():
            for subgroup_id, instance in group_instances.items():
                function = instance["function"]

                configuration_space.add_configuration_space(
                    configuration_space=function._create_config_space(),
                    prefix=f"{group_id}:{subgroup_id}",
                    delimiter=":",
                )

        configuration_space.add(
            CategoricalHyperparameter(
                name="group", choices=range(self.groups), default_value=0
            )
        )
        configuration_space.add(
            CategoricalHyperparameter(
                name="subgroup",
                choices=range(self.subgroups_per_group),
                default_value=0,
            )
        )
        return configuration_space

    def _make_hierarchical_groups(self):
        instances = {}
        x_i = 0

        # Generate seeds for each subgroup
        subgroup_seeds = self.rng.integers(low=0, high=1000, size=self.total_subgroups)
        # Create groups and subgroups
        for group_id in range(self.groups):
            group_instances = {}
            for subgroup_id in range(self.subgroups_per_group):
                subgroup_index = group_id * self.subgroups_per_group + subgroup_id
                group_instances[subgroup_id] = {
                    "function": SumOfQ(
                        seed=subgroup_seeds[subgroup_index],
                        dim=self.all_subgroup_dims[group_id][subgroup_id],
                    ),
                    "starts_at": x_i,
                }
                x_i += self.all_subgroup_dims[group_id][subgroup_id]
            instances[group_id] = group_instances
        return instances

    def _function(self, x: np.ndarray) -> float:
        """
        Evaluate the function at the given input `x`.

        Args:
            x (np.ndarray): The input vector of dimension `dim + 2`, where the last two
                entries are categorical.

        Returns:
            float: The function value at `x`.
        """
        print("x", x)
        group = x[-2]
        subgroup = x[-1]

        # Select the correct instance based on group and subgroup
        instance = self.instances[group][subgroup]["function"]
        starts_at = self.instances[group][subgroup]["starts_at"]

        x = x[starts_at : starts_at + instance.dim]
        return instance._function(x)

    def _calculate_x_min(self):
        """
        Find the global minimum across all groups and subgroups,
        and return the corresponding x_min
        and the group/subgroup that leads to it.
        """
        # Initialize a list of minimum values for each subgroup
        x_min_init = [None] * self.dim_without_categorical

        # Initialize the best known function value as infinity
        current_best_f = math.inf
        group_of_x_min = None
        subgroup_of_x_min = None

        # Iterate over all groups and subgroups
        for group_id, group_instances in self.instances.items():
            for subgroup_id, func in group_instances.items():
                # Check if this subgroup's function has a smaller f_min
                if func["function"].f_min < current_best_f:
                    # Update the best function value
                    current_best_f = func["function"].f_min

                    # Store the group and subgroup indices
                    group_of_x_min = group_id
                    subgroup_of_x_min = subgroup_id

                    # Copy the x_min of the function corresponding to this subgroup
                    x_min = [float(val) for val in func["function"].x_min]

        # Fill the global x_min array with the values of the current x_min
        instance = self.instances[group_of_x_min][subgroup_of_x_min]["function"]
        starts_at = self.instances[group_of_x_min][subgroup_of_x_min]["starts_at"]

        x_min_init[starts_at : starts_at + instance.dim] = x_min

        # Add the group and subgroup identifiers to the end of the x_min array
        x_min = [*x_min_init, group_of_x_min, subgroup_of_x_min]

        # Store the final x_min
        self._x_min = x_min

    @property
    def x_min(self) -> np.ndarray | None:
        """
        Return the cached x_min or calculate it if not already done.

        Returns:
            np.ndarray | None: The global minimum location (x_min) as a numpy array.
        """
        if self._x_min is None:
            self._calculate_x_min()

        return np.array(self._x_min)

    @property
    def f_min(self) -> float:
        """
        Get the global minimum function value.

        Returns:
            float: The global minimum value.
        """
        return self._function(self.x_min)


class InvalidParameterization(AbstractFunction):
    """
    Invalid Parameterization - C7
    This benchmark simulates a scenario where certain regions of the parameter search space
    are invalid, and attempts to evaluate the objective function in those regions will raise
    an exception.

    This benchmark represents situations where invalid parameterizations occur, such as
    forbidden combinations of parameter values, configurations that do not fit into memory,
    or situations that result in an exception or crash during algorithm execution.
    The optimizer is tasked with finding valid configurations while avoiding these invalid
    subspaces.
    """

    def __init__(
        self,
        dim: int,
        cube_size: float,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            cube_size (float): The percentage of each parameters' domain that will return
                an invalid parameterization. Must be between 0.0 and 1.0.
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )
        self.benchmark_name = "c7"

        self.rng = np.random.default_rng(seed=seed)

        self.instance = SumOfQ(seed=seed, dim=dim)

        self._configspace = self.instance._create_config_space()

        self.cube, self.cube_sides = self._make_hypercube(cube_size)
        self._x_min = None
        self.x_min_computed = False

    def _make_hypercube(self, cube_size):
        """
        Creates an invalid hypercube within the parameter space. The size of the hypercube
        is determined by the `cube_size` parameter, and its coordinates are randomly sampled
        based on the parameter bounds of the underlying quadratic function.

        Args:
            cube_size (float): The size of the invalid hypercube within the search space.

        Returns:
            tuple: A tuple containing the coordinates of the lower bounds of the hypercube
                (`cube`) and the side lengths (`cube_sides`).
        """
        cube_side_sizes = np.array(
            [
                (abs(self.instance.lower_bounds[i]) + abs(self.instance.upper_bounds[i]))
                * cube_size
                for i in range(self.dim)
            ]
        )
        cube = np.array(
            self.rng.uniform(
                low=self.instance.lower_bounds,
                high=(self.instance.upper_bounds - cube_side_sizes),
                size=self.dim,
            )
        )
        return cube, cube_side_sizes

    def _function(self, x: np.ndarray) -> float:
        invalid = self._check_hypercube(x=x)
        if not invalid:
            return self.instance._function(x)

        err_message = "Received invalid parameter value for "
        for i in invalid:
            err_message += f"x_{i}: {x[i]}, "

        raise Exception(err_message[:-1])

    def _check_hypercube(self, x: np.ndarray):
        invalid = []
        for i in range(len(x)):
            if self.cube[i] < x[i] < self.cube[i] + self.cube_sides[i]:
                invalid.append(i)
        return invalid

    @property
    def x_min(self) -> np.ndarray | None:
        """
        Evaluates the minimal value of this benchmark's function. If the minimum x value of any
        of the parameters lie withing the hypercube, that parameter's quadratic function is
        evaluated at its bounds and at the edges of the hypercube to find the smallest value
        where x is a valid parameterisation.
        """
        if self.x_min_computed:
            return self._x_min

        x_min = self.instance.x_min
        invalid = self._check_hypercube(x=x_min)
        for i in invalid:
            b = [
                self.instance.lower_bounds[i],
                self.cube[i],
                self.cube[i] + self.cube_sides[i],
                self.instance.upper_bounds[i],
            ]
            argmin = np.argmin(
                self.instance.get_value_in_single_dim(b[0]),
                self.instance.get_value_in_single_dim(b[1]),
                self.instance.get_value_in_single_dim(b[2]),
                self.instance.get_value_in_single_dim(b[3]),
            )
            x_min[i] = b[argmin]

        self._x_min = x_min
        self.x_min_computed = True

        return self._x_min

    @property
    def f_min(self) -> float:
        return self.instance._function(self.x_min)


class MixedDomains(AbstractFunction):
    """
    Mixed Domains - C8
    A synthetic benchmark function where the search space comprises domains of different sizes,
    combining dim/2 many narrow [-1; 1] and wide [-10000; 10000] ranges each. The
    objective function is simply an instance of the SumOfQ function, with these custom domains.
    """

    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        instance_parameter: float = 0.0,
        loggers: list | None = None,
    ) -> None:
        """
        Initializes the benchmark.

        Args:
            dim (int): The number of parameters to be optimized (dimensions).
            seed (int | None, optional): Random seed for reproducibility. Default is None.
            loggers (list | None, optional): List of logger objects. Default is None.
        """
        super().__init__(
            seed=seed, dim=dim, instance_parameter=instance_parameter, loggers=loggers
        )

        self.rng = np.random.default_rng(seed=seed)

        lower_bounds = [-1] * math.ceil(self.dim / 2) + [-10000] * (self.dim // 2)
        upper_bounds = [-b for b in lower_bounds]
        self.instance = SumOfQ(
            seed=seed, dim=dim, lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )

        self._configspace = self._create_config_space()

        self.benchmark_name = "c8"

    def _create_config_space(self):
        return ConfigurationSpace(
            {
                f"x_{i}": Float(
                    bounds=(
                        self.instance.lower_bounds[i],
                        self.instance.upper_bounds[i],
                    ),
                    default=0,
                    name=f"x_{i}",
                )
                for i in range(self.dim)
            },
            seed=self.seed,
        )

    def _function(self, x: ndarray) -> float:
        return self.instance._function(x=x)

    @property
    def x_min(self) -> np.ndarray | None:
        return self.instance.x_min

    @property
    def f_min(self) -> float:
        return self.instance.f_min
