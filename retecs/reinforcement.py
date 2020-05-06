from classifier import CIClassifier
import unittest
import copy


class Reinforcement:

    ALPHA = 0.1
    BETA = 0.1
    NU = 5
    EPSILON_0 = 0.01 # <---------------------- adapt later tests written for 10

    def __init__(self):
        pass

    def reinforce(self, action_set, discounted_reward):
        '''
        Adapts prediction, error, experience and fitness according to the newest discounted_reward.

        :param action_set: list of classifiers.
        :param discounted_reward: last reward - gamma * maximum(predicition_array)
        '''
        number_classifiers = sum(list(map(lambda x: x.numerosity, action_set)))
        for classifier in action_set:
            classifier.experience += 1
            # update prediction
            if classifier.experience < (1.0 / Reinforcement.BETA):
                classifier.prediction += (discounted_reward - classifier.prediction) / classifier.experience
            else:
                classifier.prediction += Reinforcement.BETA * (discounted_reward - classifier.prediction)
            # update prediction error
            absolute_error = abs(discounted_reward - classifier.prediction)
            if classifier.experience < (1.0 / Reinforcement.BETA):
                classifier.epsilon += (absolute_error - classifier.epsilon) / classifier.experience
            else:
                classifier.epsilon += Reinforcement.BETA * (absolute_error - classifier.epsilon)
            # update action set size estimate
            if classifier.experience < (1.0 / Reinforcement.BETA):
                classifier.action_set_size += (number_classifiers - classifier.action_set_size) / classifier.experience
            else:
                classifier.action_set_size += Reinforcement.BETA * (number_classifiers - classifier.action_set_size)
        self.update_fitness(action_set)

    def update_fitness(self, action_set):
        '''
        Adapts the fitness according to the accuracy (measure based on the error epsilon).

        :param action_set: list of classifiers
        '''
        accuracy_sum = 0
        accuracy_vector = {}
        for classifier in action_set:
            if classifier.epsilon < Reinforcement.EPSILON_0:
                accuracy_vector[classifier] = 1
            else:
                accuracy_vector[classifier] = Reinforcement.ALPHA * ((classifier.epsilon / Reinforcement.EPSILON_0) ** (-1.0 * Reinforcement.NU))
            accuracy_sum += accuracy_vector[classifier] * classifier.numerosity
        for classifier in action_set:
            classifier.fitness += Reinforcement.BETA * ((accuracy_vector[classifier] * classifier.numerosity) / accuracy_sum - classifier.fitness)


class Test_CIActionSelection(unittest.TestCase):

    def test_update_fitness(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.fitness = 7.0
        classifier.epsilon = 5.0
        other_classifier = copy.deepcopy(classifier)
        other_classifier.fitness = 11.0
        other_classifier.epsilon = 20.0
        other_classifier.numerosity = 2
        action_set = [classifier, other_classifier]
        r = Reinforcement()
        r.update_fitness(action_set)
        assert classifier.fitness == (7 + 0.1 * (1 / 1.00625 - 7))
        assert other_classifier.fitness == (11.0 + 0.1 * (0.003125 * 2 / 1.00625 - 11))

    def test_action_selection_small_experience(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.experience = 2.0
        classifier.prediction = 5.0
        classifier.epsilon = 3.0
        classifier.numerosity = 12.0
        classifier.action_set_size = 7.0
        action_set = [classifier]
        r = Reinforcement()
        r.reinforce(action_set, 10)
        absolute_error = abs(10 - classifier.prediction)
        assert classifier.experience == 3.0
        assert classifier.prediction == (5 + 5 / 3.0)
        assert classifier.epsilon == (3.0 + (absolute_error - 3.0) / 3.0)
        assert classifier.action_set_size == (7.0 + 5.0 / 3)

    def test_action_selection_large_experience(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.experience = 999.0
        classifier.prediction = 5.0
        classifier.epsilon = 3.0
        classifier.numerosity = 12.0
        classifier.action_set_size = 7.0
        action_set = [classifier, copy.deepcopy(classifier)]
        r = Reinforcement()
        r.reinforce(action_set, 10)
        absolute_error = abs(10 - classifier.prediction)
        assert classifier.experience == 1000.0
        assert classifier.prediction == 5.5
        assert classifier.epsilon == (3 + 0.1 * (absolute_error - 3.0))
        assert classifier.action_set_size  == (8.7)


if "__main__" == __name__:

    unittest.main()

