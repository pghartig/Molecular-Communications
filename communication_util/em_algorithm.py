
import numpy as np

def em_gausian(num_gaussians, data,iterations):
    """
    implementation of EM for gaussian model
    :param num_gaussians:
    :param data:
    :param iterations:
    :return:
    """
    num_observations = data.shape[0]
    mu = np.zeros((num_gaussians, 1))
    sigma_square = np.zeros((num_gaussians, 1))

    # current probability of each component (initialize as equiprobable?)
    alpha = np.zeros((num_gaussians, 1))
    # For each data point attribute a probability of originating from each gaussian component of the mixture.
    weights = np.zeros((num_observations, num_gaussians))

    for iteration in range(iterations):
        """
        Expectation step
        """
        # Find probability of a certain observation originating from each source using current parameters
        # TODO look for faster way to update these
        for i in num_observations:
            probabilities = probability_from_gaussian_sources(data[data[i, :]], mu, sigma_square)
            weighted_probabilities = alpha*probabilities
            weights[i, :] = weighted_probabilities/np.sum(weighted_probabilities)
        """
        Maximization step
        """
        alpha = np.sum(weights, axis=0)/num_observations
        mu = np.dot(weights.T, data)/np.sum(weights, axis=0)

    return mu,sigma_square

def probability_from_gaussian_sources(data_point, mu, sigma_square):
    """
    return (as a vector) the probability of data point originating from all parameterized distrubutions
    :param data_point:
    :param mu:
    :param sigma_square:
    :return:
    """
    return np.exp((-np.power((data_point-mu), 2))/(2*sigma_square))/np.sqrt((2*np.pi*sigma_square))

