import numpy as np
import pickle
import os

def em_gausian(num_gaussians, data, iterations, test_data= None, save= False, model=False,both=False):
    """
    implementation of EM for gaussian model
    :param num_gaussians:
    :param data:
    :param iterations:
    :return:
    """
    # Currently just initializing with some of the input data
    test_set_probability = None
    num_observations = data.shape[0]
    initialization_sample = data[0:num_gaussians]
    mu = initialization_sample
    #decide how to initialize
    sigma_square = np.ones((num_gaussians, 1))*0.1


    # current probability of each component (initialize as equiprobable)
    alpha = np.ones((num_gaussians, 1)) * (1 / num_gaussians)
    # For each data point attribute a probability of originating from each gaussian component of the mixture.
    weights = np.ones((num_observations, num_gaussians)) * (1 / num_gaussians)

    # Collect the probability of the training data set after each iteration and the difference between subsequent
    # iterations to ensure it is monotonically increasing

    likelihood_vector = []
    test_likelihood_vector = []

    for iteration in range(iterations):


        """
        Expectation step
        """
        itr_total_sequence_probability = []
        for ind, observation in enumerate(data):
            probabilities = probability_from_gaussian_sources(observation, mu, sigma_square)
            weighted_probabilities = alpha * probabilities
            new_weights = weighted_probabilities / np.sum(weighted_probabilities)
            weights[ind, :] = np.reshape(new_weights, weights[ind, :].shape)
            # This is a lower bound on the probability of this observation given the current
            test = np.log(np.sum(probabilities*alpha))
            itr_total_sequence_probability.append(test)


        if test_data is not None:
            test_obs = test_data.shape[0]
            test_set_probability = []
            for i in range(test_obs):
                test_probabilities = probability_from_gaussian_sources(test_data[i], mu, sigma_square)
                test = np.log(np.sum(test_probabilities*alpha))
                test_set_probability.append(test)
            test_likelihood_vector.append(np.sum(test_set_probability) / num_observations)

        # after each iteration, this value should become less negative and approach zero.
        likelihood_vector.append(np.sum(itr_total_sequence_probability)/num_observations)


        """
        Maximization step
        """

        alpha = np.reshape(np.sum(weights, axis=0) / num_observations, alpha.shape)
        mu = np.sum(weights.T * data, axis=1)/np.sum(weights, axis=0)
        for i in range(num_gaussians):
            sub = mu[i]
            squared_difference = np.power(data - sub, 2)
            data_weight = weights[:, i]
            total_square_difference = np.dot(squared_difference.T, data_weight)
            sigma_square[i] = total_square_difference / np.sum(data_weight)

    if save is True:
        # path = "/Users/peterhartig/Documents/Projects/moco_project/molecular-communications-project/Output/mm.pickle"
        path = "Output/mm.pickle"
        pickle_out = open(path, "wb")
        pickle.dump([mu, sigma_square, alpha], pickle_out)
        pickle_out.close()

    if model==True:
        return mixture_model(mu, sigma_square, alpha)
    if both==True:
        return mixture_model(mu, sigma_square, alpha), mu, sigma_square, alpha, likelihood_vector, test_likelihood_vector
    else:
        return mu, sigma_square, alpha, likelihood_vector, test_set_probability

def probability_from_gaussian_sources(data_point, mu, sigma_square):
    """
    return (as a vector) the probability of data point originating from all parameterized distrubutions
    :param data_point:
    :param mu:
    :param sigma_square:
    :return:
    """
    if not isinstance(mu,np.ndarray):
        return np.divide(
            np.exp(np.divide(-np.power((data_point - mu), 2), (2 * sigma_square))),
            np.sqrt(2 * np.pi * sigma_square))
    else:
        probabilities = []
        for ind, mu_i in enumerate(mu):
            probabilities.append(np.divide(
                np.exp(np.divide(-np.power((data_point - mu_i), 2), (2 * sigma_square[ind]))),
                np.sqrt(2 * np.pi * sigma_square[ind])))
        return np.asarray(probabilities)

def receive_probability(symbol,mu,sigma_square):
    return np.prod(probability_from_gaussian_sources(symbol,mu,sigma_square))

class mixture_model():
    def __init__(self, mu, sigma_square, alpha):
        self.mu = mu
        self.sigma_square = sigma_square
        self.alpha = alpha
    def get_probability(self,symbol):
        return np.sum(np.dot(self.alpha, probability_from_gaussian_sources(symbol, self.mu, self.sigma_square).T))


