import trieste
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from trieste.models.gpflow.models import GaussianProcessRegression
from docs.notebooks.util.plotting import plot_bo_points, plot_function_2d
import gpflow


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(variance), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


def excursion_probability(x, model, threshold=80.):
    mean, variance = model.model.predict_f(x)
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    threshold = tf.cast(threshold, x.dtype)
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)

def plot_excursion_probability(title, model, threshold=80.0):

    def objective_function(x):
        p = excursion_probability(x, model, threshold)
        return p * (1. - p)

    _, ax = plot_function_2d(
        objective_function,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        grid_density=70,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=[title],
        xlabel="$X_1$",
        ylabel="$X_2$",
        fill=True,
    )
    return ax


np.random.seed(1793)
tf.random.set_seed(1793)

search_space = BRANIN_SEARCH_SPACE
threshold = 80.0
observer = trieste.objectives.utils.mk_observer(branin)

num_initial_points = 11
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)
model = build_model(initial_data)



# fitting the model only to the initial data
initial_model = build_model(initial_data)
initial_model.optimize(initial_data)


def value_function(x):
    p = excursion_probability(x, model, threshold)
    return p * (1. - p)

num_centroids = 100
num_iterations = 20000
centroids =  search_space.sample_halton(num_centroids).numpy()
centroids_value = value_function(centroids).numpy()
counter = np.ones([num_centroids, ])

for j in range(num_iterations):
    point = search_space.sample(1).numpy()
    diff = centroids - np.repeat(point, num_centroids, axis=0)
    dist_to_centroids = np.sqrt(np.sum(diff ** 2, axis = -1))
    i_closest = np.argmin(dist_to_centroids * centroids_value.flatten())
    # i_closest = np.argmin(dist_to_centroids)
    point_value =  value_function(point).numpy()

    # weight_centroid = counter[i_closest] * centroids_value[i_closest]
    # weight_point = point_value
    # centroids[i_closest, :] = (weight_centroid * centroids[i_closest, :] + weight_point * point) / \
    #                        (weight_centroid + weight_point)

    centroids[i_closest, :] = (counter[i_closest] * centroids[i_closest,:] + point) / \
                              (counter[i_closest]  + 1)

    counter[i_closest] += 1

    centroids_value[i_closest] = value_function(centroids[i_closest, :].reshape(1, -1)).numpy()


ax = plot_excursion_probability(
    "Probability of excursion, initial data",
    initial_model,
)
ax[0,0].scatter(centroids[:, 0], centroids[:, 1])

print(counter)
