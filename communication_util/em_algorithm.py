
import numpy as np

def em_gausian(num_gaussians, data,iterations):
    """
    implementation of EM for gaussian model
    :param num_gaussians:
    :param data:
    :param iterations:
    :return:
    """
    # TODO look at how to initialize
    num_observations = data.shape[0]
    initialization_sample = data[0:num_gaussians]
    mu = initialization_sample
    sigma_square = np.ones((num_gaussians, 1))*np.var(initialization_sample)

    # current probability of each component (initialize as equiprobable)
    alpha = np.ones((num_gaussians, 1))*(1/num_gaussians)
    # For each data point attribute a probability of originating from each gaussian component of the mixture.
    weights = np.ones((num_observations, num_gaussians))*(1/num_gaussians)

    for iteration in range(iterations):
        """
        Expectation step
        """
        # Find probability of a certain observation originating from each source using current parameters
        # TODO look for faster way to update these
        for i in range(num_observations):
            probabilities = probability_from_gaussian_sources(data[i], mu, sigma_square)
            weighted_probabilities = alpha*probabilities
            new_weights = weighted_probabilities/np.sum(weighted_probabilities)
            weights[i, :] = np.reshape(new_weights, weights[i, :].shape)
            # for j in range(mu.size):
            #     weights[i, j] = new_weights[j]

        alpha = np.reshape(np.sum(weights, axis=0)/num_observations, alpha.shape)

        """
        Maximization step
        """
        mu = np.dot(weights.T, data)/np.sum(weights, axis=0)
        for i in range(num_gaussians):
            sigma_square[i] = np.dot(weights[:, i].T, np.power(data-mu[i], 2))/np.sum(weights[:, i], axis=0)

    return mu, sigma_square

def probability_from_gaussian_sources(data_point, mu, sigma_square):
    """
    return (as a vector) the probability of data point originating from all parameterized distrubutions
    :param data_point:
    :param mu:
    :param sigma_square:
    :return:
    """
    test1 = np.zeros(sigma_square.shape)
    for i in range(mu.size):
        test1[i] = np.divide(np.exp(np.divide(-np.power(data_point - mu[i], 2), 2 * sigma_square[i])), np.sqrt(2*np.pi*sigma_square[i]))
    return test1

