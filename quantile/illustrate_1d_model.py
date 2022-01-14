import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
from trieste.data import Dataset
from model_utils import build_hetgp_rff_model, ASymmetricLaplace
from inducing_point_selector import KMeans, GridSampler

np.random.seed(1789)
tf.random.set_seed(1789)

quantile_level = 0.9

def noisefree_fun(x):
    return .05 *tf.sin(x * np.pi * 2.) + .05 * x

def fun(x):
    x = 1.3 * x
    # eps_left = tf.random.uniform(x.shape, -.5, .5, dtype=tf.float64)

    eps_center = tfp.distributions.LogNormal(tf.cast(0., dtype=tf.float64),
                                             tf.cast(1., dtype=tf.float64)).sample(x.shape)

    eps_right = tf.random.normal(x.shape, 0., 1., dtype=tf.float64)

    # eps_left = 3. * eps_left * tf.maximum(0.2 - x, 0)

    soft_left = 1. - tfp.distributions.Normal(tf.cast(0.3, dtype=tf.float64),
                                         tf.cast(.1, dtype=tf.float64)).cdf(x)

    soft_right = tfp.distributions.Normal(tf.cast(1.1, dtype=tf.float64),
                                          tf.cast(.1, dtype=tf.float64)).cdf(x)

    eps_center = .1 * eps_center * soft_left  # norm.pdf(x - 0.7, 0, 0.08)
    eps_right = .2 * eps_right * soft_right  # tf.maximum(0, x - .8) ** 2

    return noisefree_fun(x) + eps_center + eps_right

def quantile_fun(x, quantile_level):
    x = 1.3 * x
    # q_left = tfp.distributions.Uniform(tf.cast(-.5, dtype=tf.float64),
    #                                          tf.cast(.5, dtype=tf.float64)).quantile(quantile_level)
    q_center = tfp.distributions.LogNormal(tf.cast(0., dtype=tf.float64),
                                             tf.cast(1., dtype=tf.float64)).quantile(quantile_level)
    q_right = tfp.distributions.Normal(tf.cast(0., dtype=tf.float64),
                                             tf.cast(1., dtype=tf.float64)).quantile(quantile_level)

    soft_left = 1. - tfp.distributions.Normal(tf.cast(0.3, dtype=tf.float64),
                                             tf.cast(.1, dtype=tf.float64)).cdf(x)

    soft_right = tfp.distributions.Normal(tf.cast(1.1, dtype=tf.float64),
                                             tf.cast(.1, dtype=tf.float64)).cdf(x)

    # q_left = 3. * q_left * tf.maximum(0.2 - x, 0)
    q_center = .1 * q_center * soft_left  #norm.pdf(x - 0.7, 0, 0.08)
    q_right = .2 * q_right * soft_right  #tf.maximum(0, x - .8) ** 2

    return noisefree_fun(x) + q_center + q_right


# Set up problem
observer = lambda qp: Dataset(qp, fun(qp))
search_space = trieste.space.Box([0.], [1.])
initial_query_points = search_space.sample(500)
data = observer(initial_query_points)


# Fit heterogeneous quantile model
qhet_model = build_hetgp_rff_model(data=data,
                                     num_features=1000,
                                     likelihood_distribution=
                                     lambda loc, scale: ASymmetricLaplace(loc, scale, tau=quantile_level),
                                     num_inducing_points=40,
                                     inducing_point_selector=GridSampler(search_space))
qhet_model.optimize(data)

qhet_model2 = build_hetgp_rff_model(data=data,
                                     num_features=1000,
                                     likelihood_distribution=
                                     lambda loc, scale: ASymmetricLaplace(loc, scale, tau=1. - quantile_level),
                                     num_inducing_points=40,
                                     inducing_point_selector=GridSampler(search_space))
qhet_model2.optimize(data)

# Fit hetGP model
hetgp = build_hetgp_rff_model(data=data,
                              num_features=1000,
                              likelihood_distribution=tfp.distributions.Normal,
                              num_inducing_points=40,
                              inducing_point_selector=GridSampler(search_space))
hetgp.optimize(data)

# Fit homogeneous quantile model
qhom_model = build_hetgp_rff_model(data=data,
                                     num_features=1000,
                                     likelihood_distribution=
                                     lambda loc, scale: ASymmetricLaplace(loc, scale, tau=quantile_level),
                                     num_inducing_points=40,
                                     inducing_point_selector=GridSampler(search_space),
                                homogeneous=True)
qhom_model.optimize(data)

qhom_model2 = build_hetgp_rff_model(data=data,
                                     num_features=1000,
                                     likelihood_distribution=
                                     lambda loc, scale: ASymmetricLaplace(loc, scale, tau=1. - quantile_level),
                                     num_inducing_points=40,
                                     inducing_point_selector=GridSampler(search_space),
                                homogeneous=True)
qhom_model2.optimize(data)

x = np.linspace(0., 1., 1000)
x = tf.constant(x.reshape(-1, 1), dtype=tf.float64)
f = fun(x)

nrows = 2
ncols = 4
fig, ax = plt.subplots(nrows, ncols, sharex='all', sharey='all')

q_levels = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95]
for ql in q_levels:
    q = quantile_fun(x, ql)
    ax[0,0].plot(x.numpy(), q.numpy())
    ax[1, 0].plot(x.numpy(), q.numpy())

q_act1 = quantile_fun(x, quantile_level)
q_act2 = quantile_fun(x, 1. - quantile_level)
q_act = tf.concat([q_act1, q_act2], axis=-1)
for row in range(nrows):
    for col in range(ncols):
        ax[row, col].plot(x.numpy(), q_act[:, row].numpy(), color="red", linewidth=3)
        ax[row, col].scatter(data.query_points.numpy(), data.observations.numpy(), color="black")

# hetQuantile plot
qhet_mean, qhet_var = qhet_model.predict(x)
ax[0,1].plot(x.numpy(), qhet_mean[:, 0].numpy(), color="green", linewidth=3)
y_up = qhet_mean[:, 0].numpy() + 1.96 * np.sqrt(qhet_var[:, 0].numpy())
y_lo = qhet_mean[:, 0].numpy() - 1.96 * np.sqrt(qhet_var[:, 0].numpy())
ax[0,1].fill_between(x.numpy().flatten(), y_up.flatten(), y_lo.flatten(), alpha=0.3, color="green")

qhet_mean, qhet_var = qhet_model2.predict(x)
ax[1,1].plot(x.numpy(), qhet_mean[:, 0].numpy(), color="green", linewidth=3)
y_up = qhet_mean[:, 0].numpy() + 1.96 * np.sqrt(qhet_var[:, 0].numpy())
y_lo = qhet_mean[:, 0].numpy() - 1.96 * np.sqrt(qhet_var[:, 0].numpy())
ax[1,1].fill_between(x.numpy().flatten(), y_up.flatten(), y_lo.flatten(), alpha=0.3, color="green")

# hetGP plot
hetgp_mean, hetgp_var = hetgp.predict(x)
lik_layer = hetgp.model_gpflux.likelihood_layer
dist = lik_layer.likelihood.conditional_distribution(hetgp_mean)
hetgp_qpred = dist.quantile(value=quantile_level)
ax[0, 2].plot(x.numpy(), hetgp_qpred[:, 0].numpy(), color="orange", linewidth=3)

def quantile_traj_from_hetgp(model, at, qlevel):
    trajectory = model.sample_trajectory()
    lik_layer = model.model_gpflux.likelihood_layer
    dist = lik_layer.likelihood.conditional_distribution(trajectory(at))
    return dist.quantile(value=qlevel)

for i in range(10):
    hetgp_traj = quantile_traj_from_hetgp(hetgp, x, quantile_level)
    ax[0,2].plot(x.numpy(), hetgp_traj[:, 0].numpy(), color="orange", linewidth=1)
    hetgp_traj = quantile_traj_from_hetgp(hetgp, x, 1. - quantile_level)
    ax[1, 2].plot(x.numpy(), hetgp_traj[:, 0].numpy(), color="orange", linewidth=1)


# homGP plot
qhom_mean, qhom_var = qhom_model.predict(x)
ax[0,3].plot(x.numpy(), qhom_mean[:, 0].numpy(), color="blue", linewidth=3)
y_up = qhom_mean[:, 0].numpy() + 1.96 * np.sqrt(qhom_var[:, 0].numpy())
y_lo = qhom_mean[:, 0].numpy() - 1.96 * np.sqrt(qhom_var[:, 0].numpy())
ax[0,3].fill_between(x.numpy().flatten(), y_up.flatten(), y_lo.flatten(), alpha=0.3, color="blue")

qhom_mean, qhom_var = qhom_model2.predict(x)
ax[1,3].plot(x.numpy(), qhom_mean[:, 0].numpy(), color="blue", linewidth=3)
y_up = qhom_mean[:, 0].numpy() + 1.96 * np.sqrt(qhom_var[:, 0].numpy())
y_lo = qhom_mean[:, 0].numpy() - 1.96 * np.sqrt(qhom_var[:, 0].numpy())
ax[1,3].fill_between(x.numpy().flatten(), y_up.flatten(), y_lo.flatten(), alpha=0.3, color="blue")

for row in range(nrows):
    ax[row, 0].set_ylabel("y")

for col in range(ncols):
    ax[1, col].set_xlabel("x")

fig.tight_layout()

ax[0, 0].set_title("Actual quantiles")
ax[0, 1].set_title("Quantile hetGP")
ax[0, 2].set_title("Gaussian hetGP")
ax[0, 3].set_title("Gaussian homGP")
