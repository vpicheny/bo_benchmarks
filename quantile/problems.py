import tensorflow as tf
import tensorflow_probability as tfp
from trieste.objectives import scaled_branin, hartmann_3, SCALED_BRANIN_MINIMUM
from trieste.space import Box
from scipy.stats import norm

from trieste.acquisition.optimizer import automatic_optimizer_selector, generate_continuous_optimizer
from trieste.acquisition import AcquisitionFunction
from trieste.space import Box

from gld_problems import create_gld_trajectory


class Problem:
    fun = None
    quantile_fun = None
    dim:int
    lower_bounds = [0, 0]
    upper_bounds = [1, 1]
    quantile_level:float
    minimum:float


class DummyAcquisition(AcquisitionFunction):

    def __init__(self, fun):
        self._fun = fun

    def __call__(self, x):
        return -self._fun(tf.squeeze(x, -2))


def get_problem(problem_specs: [str, int, int, float]):

    name, seed, input_dim, quantile_level = problem_specs

    problem = Problem
    if name == "gauss_noise_branin":
        noise = .1

        beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=quantile_level).numpy()

        def noise_sd(x):
            return noise * tf.reduce_sum(x, axis=-1, keepdims=True)

        def fun(x):
            y = scaled_branin(x)
            return y + tf.random.normal(y.shape, stddev=noise_sd(x), dtype=y.dtype)

        def quantile_fun(x):
            y = scaled_branin(x)
            return y + beta * noise_sd(x)

        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.lower_bounds = [0., 0.]
        problem.upper_bounds = [1., 1.]
        problem.quantile_level = quantile_level
        problem.dim = 2
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 1000000)

        return problem

    elif name == "exp_noise_branin":
        noise = .1

        beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=quantile_level).numpy()

        def noise_sd(x):
            return noise * tf.reduce_sum(x, axis=-1, keepdims=True)

        def fun(x):
            y = scaled_branin(x)
            return y + tf.exp(tf.random.normal(y.shape, stddev=noise_sd(x), dtype=y.dtype))

        def quantile_fun(x):
            y = scaled_branin(x)
            return y + tf.exp(beta * noise_sd(x))

        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.lower_bounds = [0., 0.]
        problem.upper_bounds = [1., 1.]
        problem.quantile_level = quantile_level
        problem.dim = 2
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 1000000)

        return problem

    elif name == "hartmann_3":
        noise = .1

        beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=quantile_level).numpy()

        def noise_sd(x):
            return noise * (4. * tf.sin(x[:, 0:1]) + tf.cos(3. * x[:, 1:2]) +
                            tf.cos(tf.reduce_sum(x, axis=-1, keepdims=True))/ 2.) ** 2

        def fun(x):
            y = hartmann_3(x)
            return y + tf.random.normal(y.shape, stddev=noise_sd(x), dtype=y.dtype)

        def quantile_fun(x):
            y = hartmann_3(x)
            return y + beta * noise_sd(x)

        problem.lower_bounds = [0., 0., 0.]
        problem.upper_bounds = [1., 1., 1.]
        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.quantile_level = quantile_level
        problem.dim = 3
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 1000000)
        return problem

    elif name == "flat_branin_noise":

        beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=quantile_level).numpy()

        def noise_sd(x):
            return (scaled_branin(x) - SCALED_BRANIN_MINIMUM) + 0.1

        def fun(x):
            zeros = tf.zeros_like(x[:, 0:1])
            return zeros + tf.random.normal(zeros.shape, stddev=noise_sd(x), dtype=x.dtype)

        def quantile_fun(x):
            return beta * noise_sd(x)

        problem.lower_bounds = [0., 0.]
        problem.upper_bounds = [1., 1.]
        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.quantile_level = quantile_level
        problem.dim = 2
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 1000000)
        return problem

    elif name == "1d":

        def noisefree_fun(x):
            return tf.sin(x * 3.14 * 2.) + .25 * x

        def fun(x):
            eps_left = tf.random.uniform(x.shape, -.5, .5)
            eps_center = tfp.distributions.LogNormal(0., 1.).sample(x.shape)
            eps_right = tf.random.normal(x.shape, 0., 1.)

            eps_left = 2. * eps_left * tf.maximum(0.4 - x, 0)
            eps_center = .05 * eps_center * norm.pdf(x - 0.5, 0, 0.05)
            eps_right = eps_right * tf.maximum(0, x - 0.6)

            return noisefree_fun(x) + eps_left + eps_center + eps_right

        def quantile_fun(x):
            q_left = tfp.distributions.Uniform(-.5, .5).quantile(quantile_level)
            q_center = tfp.distributions.LogNormal(0., 1.).quantile(quantile_level)
            q_right = tfp.distributions.Normal(0., 1.).quantile(quantile_level)

            q_left = 2. * q_left * tf.maximum(0.4 - x, 0)
            q_center = .05 * q_center * norm.pdf(x - 0.5, 0, 0.05)
            q_right = q_right * tf.maximum(0, x - 0.6)

            return noisefree_fun(x) + q_left + q_center + q_right

        problem.lower_bounds = 0.
        problem.upper_bounds = 1.
        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.quantile_level = quantile_level
        problem.dim = 1
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 100000)
        return problem

    elif name == "gld":

        if input_dim == 6:
            lengthscale=1.
        else:
            lengthscale=0.5

        fun, quantile_fun = create_gld_trajectory(input_dim=input_dim, lengthscale=lengthscale, seed=seed, tau=quantile_level)

        problem.lower_bounds = [0.] * input_dim
        problem.upper_bounds = [1.] * input_dim
        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.quantile_level = quantile_level
        problem.dim = input_dim
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 100000, 10)
        return problem


def get_minimum(fun, lb, ub, num_samples, num_batches=1):

    print("finding minimum")
    search_space = Box(lb, ub)
    acquisition_function = DummyAcquisition(fun)
    optimizer = generate_continuous_optimizer(num_initial_samples=5000, num_optimization_runs=50)
    point = optimizer(search_space, acquisition_function)
    return fun(point)


    # print("Computing min by MC")
    # current_min = 1e16
    # for _ in range(num_batches):
    #     points = Box(lb, ub).sample(num_samples)
    #     y = fun(points)
    #     new_min = tf.reduce_min(y).numpy()
    #     print(new_min)
    #     if new_min < current_min:
    #         current_min = new_min
    #
    # # return current_min
    #
    # return fun(point)
