import unittest
import copy
from classifier import CIClassifier
from matching import CIMatching
from action_selection import ActionSelection
from reinforcement import Reinforcement
from genetic_algorithm import CIGeneticAlgorithm
import random
import pickle


class XCS:

    GAMMA = 0.71

    def __init__(self, max_population_size, possible_actions=[], histlen=42):
        self.name = "XCS"
        self.action_size = len(possible_actions)
        self.max_population_size = max_population_size
        self.possible_actions = possible_actions
        self.population = []
        self.time_stamp = 1
        self.action_history = []
        self.old_action_history = []
        self.reinforce = Reinforcement()
        self.ga = CIGeneticAlgorithm(possible_actions)
        #################################
        self.single_testcases = True
        self.histlen = histlen
        #################################
        # stuff for batch update
        self.max_prediction_sum = 0
        self.rewards = None
        self.p_explore = 0.25
        self.train_mode = True

    def get_action(self, state):
        '''
        :param state: State in Retects. In the XCS world = situation.

        :return : a action
        '''
        theta_mna = len(self.possible_actions)
        matcher = CIMatching(theta_mna, self.possible_actions)
        match_set = matcher.get_match_set(self.population, state, self.time_stamp)
        self.p_explore = (self.p_explore - 0.1) * 0.99 + 0.1
        action_selector = ActionSelection(self.possible_actions, self.p_explore)
        prediction_array = action_selector.get_prediction_array(match_set)
        action = action_selector.select_action(prediction_array, self.train_mode)
        max_val = prediction_array[action] # on policy
        #max(prediction_array.keys(), key=(lambda k: prediction_array[k]))
        action_set = action_selector.get_action_set(match_set, action)
        self.max_prediction_sum += max_val
        self.action_history.append((state, action_set))
        return action

    def reward(self, new_rewards):
        try:
            x = float(new_rewards)
            new_rewards = [x] * len(self.action_history)
        except Exception as _:
            if len(new_rewards) < len(self.action_history):
                raise Exception('Too few rewards')
        old_rewards = self.rewards
        self.rewards = new_rewards
        if old_rewards is not None:
            avg_max_pred = self.max_prediction_sum / len(self.action_history)
            for i in range(0, len(old_rewards)):
                discounted_reward = old_rewards[i] + XCS.GAMMA * avg_max_pred
                old_sigma, old_action_set = self.old_action_history[i]
                self.reinforce.reinforce(old_action_set, discounted_reward)
                self.ga.perform_iteration(old_action_set, old_sigma, self.population, self.time_stamp)
                self.time_stamp += 1
        self.max_prediction_sum = 0
        self.old_action_history = self.action_history
        self.action_history = []
        self.delete_from_population()

    def delete_from_population(self):
        '''
        Deletes as many classifiers as necessary until the population size is within the
        defined bounds.
        '''
        total_numerosity = sum(list(map(lambda x: x.numerosity, self.population)))
        while len(self.population) > self.max_population_size:
            total_fitness = sum(list(map(lambda x: x.fitness, self.population)))
            avg_fitness = total_fitness / total_numerosity
            vote_sum = sum(list(map(lambda x: x.deletion_vote(avg_fitness), self.population)))
            choice_point = random.random() * vote_sum
            vote_sum = 0
            for classifier in self.population:
                vote_sum += classifier.deletion_vote(avg_fitness)
                if vote_sum > choice_point:
                    if classifier.numerosity > 1:
                        classifier.numerosity = classifier.numerosity - 1
                    else:
                        self.population.remove(classifier)

    def save(self, filename):
        """ Stores agent as pickled file """
        pickle.dump(self, open(filename + '.p', 'wb'), 2)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


class Test_CIClassifier(unittest.TestCase):

    def test_deletion_vote(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.fitness = 10.0
        classifier.action_set_size = 10.0
        classifier.experience = 100.0
        classifier.numerosity = 1.0
        other_classifier = copy.deepcopy(classifier)
        different_classifier = copy.deepcopy(classifier)
        xcs = XCS(1)
        xcs.population = [classifier, other_classifier, different_classifier]
        xcs.delete_from_population()
        assert len(xcs.population) == 1


if "__main__" == __name__:
    unittest.main()