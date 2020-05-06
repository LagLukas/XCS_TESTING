import random
import copy
import unittest
from matching import CIMatching
from classifier import CIClassifier


class CIGeneticAlgorithm:

    THETA_GA = 25
    CHI = 0.75
    CROSS_OVER_ALPHA = 0.6
    MU = 0.025

    def __init__(self, possible_actions):
        '''
        :param possible_actions: list of actions that the agent may perform (necessary for mutation).
        '''
        self.possible_actions = possible_actions

    def perform_iteration(self, action_set, sigma, population, time_stamp):
        '''
        Performs an iteration of the genetic algorithm if a certain amount of time
        has been passed.

        :param action_set: set of classifiers matching sigma that all have the same action.
        :param sigma: CI situation.
        :param population: list of classifiers.
        :param time_stamp: discrete current time.
        '''
        classifier_count = sum(map(lambda x: x.numerosity, action_set))
        ts_n = sum(map(lambda x: x.numerosity * x.timestamp, action_set))
        if time_stamp - ts_n / classifier_count > CIGeneticAlgorithm.THETA_GA:
            for classifier in action_set:
                classifier.timestamp = time_stamp
            parent_1 = self.select_offspring(action_set)
            parent_2 = self.select_offspring(action_set)
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)
            child_1.numerosity = 1
            child_2.numerosity = 1
            child_1.experience = 0
            child_2.experience = 0
            if random.random() < CIGeneticAlgorithm.CHI:
                self.apply_crossover(child_1, child_2)
                child_1.prediction = (parent_1.prediction + parent_2.prediction) / 2.0
                child_2.prediction = child_1.prediction
                child_1.epsilon = (parent_1.epsilon + parent_2.epsilon) / 2.0
                child_2.epsilon = child_1.epsilon
                child_1.fitness = (parent_1.fitness + parent_2.fitness) / 2.0
                child_2.fitness = child_1.fitness
            child_1.fitness = 0.1 * child_1.fitness
            child_2.fitness = 0.1 * child_2.fitness
            self.mutation(child_1, sigma)
            self.mutation(child_2, sigma)
            population.append(child_1)
            population.append(child_2)
        # TODO add deletion

    def select_offspring(self, action_set):
        '''
        Performs a roulette wheel selection on the action_set.

        :param action_set: a list of classifiers.

        :return : 
        '''
        fitness_sum = sum(map(lambda x: x.fitness, action_set))
        choice_point = random.random() * fitness_sum
        fitness_sum = 0.0
        for classifier in action_set:
            fitness_sum += classifier.fitness
            if fitness_sum >= choice_point:
                return classifier

    def apply_crossover(self, classifier_1, classifier_2):
        '''
        Applies a crossover operation according to "an algorithmic description of xcs" to the ternary
        attributes of the classifiers. For the continious attributes a arthmetic crossover is used.

        :param classifier_1: a CI classifier.
        :param classifier_2: a CI classifier.
        '''
        x = int(random.random() * len(classifier_1.previous_results_conditions))
        y = int(random.random() * len(classifier_1.previous_results_conditions))
        if y > x:
            temp = x
            x = y
            y = temp
        for i in range(x, y):
            temp = classifier_1.previous_results_conditions[i]
            classifier_1.previous_results_conditions[i] = classifier_2.previous_results_conditions[i]
            classifier_2.previous_results_conditions[i] = temp
        alpha = CIGeneticAlgorithm.CROSS_OVER_ALPHA
        a = classifier_1.last_executions_condition
        b = classifier_2.last_executions_condition
        classifier_1.last_executions_condition = (alpha * a[0] + (1 - alpha) * b[0], alpha * a[1] + (1 - alpha) * b[1])
        classifier_2.last_executions_condition = ((1 - alpha) * a[0] + alpha * b[0], (1 - alpha) * a[1] + alpha * b[1])
        a = classifier_1.average_duration_condition
        b = classifier_2.average_duration_condition
        classifier_1.average_duration_condition = (alpha * a[0] + (1 - alpha) * b[0], alpha * a[1] + (1 - alpha) * b[1])
        classifier_2.average_duration_condition = ((1 - alpha) * a[0] + alpha * b[0], (1 - alpha) * a[1] + alpha * b[1])

    def mutation(self, classifier, situation):
        '''
        Performs a mutation according to "an algorithmic description of xcs" for the ternary attributes and the action.
        For the continous conditions new intervall is randomly created with a certain propability.

        :param situation: a valid CI situation
        :param classifier: a CI classifier
        '''
        for i in range(0, len(classifier.previous_results_conditions)):
            if random.random() < CIGeneticAlgorithm.MU:
                if classifier.previous_results_conditions[i] == "#":
                    classifier.previous_results_conditions[i] = situation["previous_results"][i]
                else:
                    classifier.previous_results_conditions[i] = "#"
        if random.random() < CIGeneticAlgorithm.MU:
            bound_lower = situation["last_execution"] - random.random() * CIMatching.MAX_PAST
            bound_upper = situation["last_execution"] + random.random() * CIMatching.MAX_PAST
            classifier.last_executions_condition = (bound_lower, bound_upper)
        if random.random() < CIGeneticAlgorithm.MU:
            duration = situation["duration"]
            duration_border = random.random() * CIMatching.DURATION
            classifier.average_duration_condition = (duration - duration_border, duration + duration_border)
        if random.random() < CIGeneticAlgorithm.MU:
            index = int(random.random() * len(self.possible_actions))
            classifier.action = self.possible_actions[index]


class Test_CIGeneticAlgorithm(unittest.TestCase):

    def test_selection(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.fitness = 10
        average_duration = (42, 45)
        previous_results = [True, False, "#"]
        last_executions = (0, 3)
        other_classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        other_classifier.fitness = 20
        population = [classifier, other_classifier]
        ga = CIGeneticAlgorithm(["Alfons", "Bfons"])
        res = ga.select_offspring(population)
        assert res in population

    def test_crossover(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier.fitness = 10
        average_duration = (12, 40)
        previous_results = [True, False, "#"]
        last_executions = (2, 5)
        other_classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        other_classifier.fitness = 20
        ga = CIGeneticAlgorithm(["Alfons", "Bfons"])
        ga.apply_crossover(classifier, other_classifier)
        assert classifier.last_executions_condition == (0.8, 3.8)
        assert classifier.average_duration_condition == (30, 43)
        assert other_classifier.average_duration_condition == (24, 42)
        assert (classifier.previous_results_conditions[0] == "#" or classifier.previous_results_conditions[0] == True)

    def test_mutation(self):
        situation = {}
        situation["duration"] = 43
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 2
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        ga = CIGeneticAlgorithm(["Alfons", "Bfons"])
        for _ in range(0, 10):
            ga.mutation(classifier, situation)
            assert classifier.matches(situation)


if "__main__" == __name__:
    unittest.main()
