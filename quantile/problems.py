import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from trieste.objectives import scaled_branin, hartmann_3, SCALED_BRANIN_MINIMUM
from scipy.stats import norm

from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition import AcquisitionFunction
from trieste.space import Box

from gld_problems import create_gld_trajectory
import gym
from trieste.utils import to_numpy


N_RUNS = 1


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
            lengthscale = 1.
        else:
            lengthscale = 0.5

        fun, quantile_fun = create_gld_trajectory(input_dim=input_dim, lengthscale=lengthscale, seed=seed, tau=quantile_level)

        problem.lower_bounds = [0.] * input_dim
        problem.upper_bounds = [1.] * input_dim
        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.quantile_level = quantile_level
        problem.dim = input_dim
        problem.minimum = get_minimum(problem.quantile_fun, problem.lower_bounds, problem.upper_bounds, 100000, 10)
        return problem


    elif name in ("lunar_lander_6", "lunar_lander_8", "lunar_lander_12"):

        env_name = "LunarLander-v2"
        env = gym.make(env_name)
        steps_limit = 1000
        timeout_reward = -100
        quantile_level = 0.9

        if name == "lunar_lander_6":
            problem.dim = 6
        elif name == "lunar_lander_8":
            problem.dim = 8
        elif name == "lunar_lander_8":
            problem.dim = 12

        def fun(x):
            return lander_objective(x, env, steps_limit, timeout_reward, problem.dim) / 300.

        def quantile_fun(x):
            xx = np.repeat(x, 100, axis=-2)
            ff = fun(xx)
            return np.quantile(ff, q=quantile_level, axis=-2)

        problem.lower_bounds = [0.] * problem.dim
        problem.upper_bounds = [1.2] * problem.dim
        problem.fun = fun
        problem.quantile_fun = quantile_fun
        problem.quantile_level = quantile_level
        problem.minimum = 0 # dummy value
        return problem


def lander_objective(x, env, steps_limit, timeout_reward, dim):

    if dim == 6:
        heuristic_Controller = heuristic_Controller_6d
    elif dim == 8:
        heuristic_Controller = heuristic_Controller_8d
    else:
        heuristic_Controller = heuristic_Controller_12d

    # for each point compute average reward over n_runs runs
    all_rewards = []
    for w in to_numpy(x):
        rewards = [demo_heuristic_lander(env, w, steps_limit, timeout_reward, heuristic_Controller) for _ in range(N_RUNS)]
        all_rewards.append(rewards)

    rewards_tensor = tf.convert_to_tensor(all_rewards, dtype=tf.float64)

    # triste minimizes, and we want to maximize
    return -1 * tf.reshape(tf.math.reduce_mean(rewards_tensor, axis=1), (-1, 1))


def demo_heuristic_lander(env, w, steps_limit, timeout_reward, heuristic_Controller, print_reward=False):
    total_reward = 0
    steps = 0
    s = env.reset()

    while True:
        if steps > steps_limit:
            total_reward -= timeout_reward
            break

        a = heuristic_Controller(s, w)
        s, r, done, info = env.step(a)
        total_reward += r

        steps += 1
        if done:
            break

    if print_reward:
        print(f"Total reward: {total_reward}")

    return total_reward


def heuristic_Controller_6d(s, w):  # takes 6 dim w of config
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    if angle_targ > 0.4:
        angle_targ = 0.4
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[0] - (s[5]) * w[1]
    hover_todo = (hover_targ - s[1]) * w[2] - (s[3]) * w[3]

    if s[6] or s[7]:
        angle_todo = w[4]
        hover_todo = -(s[3]) * w[5]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1
    return a


def heuristic_Controller_8d(s, w):  # takes 8 dim w of config
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = 0.0
        hover_todo = -(s[3]) * 0.5

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1
    return a


def heuristic_Controller_12d(s, w):  # takes 12 dim w of config
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


def get_minimum(fun, lb, ub, num_samples, num_batches=1):

    print("finding minimum")
    search_space = Box(lb, ub)
    acquisition_function = DummyAcquisition(fun)
    optimizer = generate_continuous_optimizer(num_initial_samples=5000, num_optimization_runs=50)
    point = optimizer(search_space, acquisition_function)
    return fun(point)