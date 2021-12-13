import tensorflow as tf
import tensorflow_probability as tfp
from trieste.objectives import scaled_branin, hartmann_3, SCALED_BRANIN_MINIMUM
from trieste.space import Box


class Problem:
    fun = None
    quantile_fun = None
    dim:int
    lower_bounds = [0, 0]
    upper_bounds = [1, 1]
    quantile_level:float
    minimum:float


def get_problem(name):
    problem = Problem
    if name == "gauss_noise_branin":
        noise = .1
        quantile_level = 0.75

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
        quantile_level = 0.9

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
        quantile_level = 0.9

        beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=quantile_level).numpy()

        def noise_sd(x):
            return noise * (4. * tf.sin(x[:, 0:1]) + tf.cos(3. * x[:, 1:2]) +
                            tf.cos(tf.reduce_sum(x, axis=-1, keepdims=True))/ 2.)

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
        quantile_level = 0.9

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


def get_minimum(fun, lb, ub, num_samples):
    points = Box(lb, ub).sample(num_samples)
    y = fun(points)
    return tf.reduce_min(y).numpy()
