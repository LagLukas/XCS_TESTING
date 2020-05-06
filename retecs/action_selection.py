import random
import unittest
from classifier import CIClassifier


class ActionSelection:

    def __init__(self, available_actions, p_exploration):
        self.p_exploration = p_exploration

    def get_prediction_array(self, match_set):
        '''
        Calculations the prediction array whose entries are actions. Their value
        indicates the likelihood of a high reward if the action will be performed.
        For this calculation only matched classifiers are considered.

        :param match_set: list of matching classifiers.

        :return : prediction array.
        '''
        self.available_actions = list(set(map(lambda x: x.action, match_set)))
        prediciton_array = {}
        fitness_sum_array = {}
        for action in self.available_actions:
            prediciton_array[action] = None
            fitness_sum_array[action] = 0
        for classifier in match_set:
            action = classifier.action
            if prediciton_array[action] is None:
                prediciton_array[action] = classifier.fitness * classifier.prediction
            else:
                prediciton_array[action] = prediciton_array[action] + classifier.fitness * classifier.prediction
            fitness_sum_array[action] += classifier.fitness
        for action in self.available_actions:
            if fitness_sum_array[action] > 0:
                prediciton_array[action] = prediciton_array[action] / fitness_sum_array[action]
        return prediciton_array

    def select_action(self, prediction_array, train_mode=True):
        '''
        Performs a epsilon greedy strategy to select an action based upon the
        prediction array.

        :param prediction_array: xcs prediction array.
        :param train_mode: true -> use best strategy and no random exploration.

        :return : a action
        '''
        if random.random() < self.p_exploration and train_mode:
            index = int(random.random() * len(self.available_actions))
            return self.available_actions[index]
        else:
            return max(prediction_array.keys(), key=(lambda k: prediction_array[k]))

    def get_action_set(self, match_set, action):
        '''
        Retrieves all classifier from the match set that have the corresponding action

        :param match_set: list of classifiers.
        :param action: action to filter by

        :return : list of classifiers who all share the same given action
        '''
        return list(filter(lambda x: x.action == action, match_set))


class Test_CIActionSelection(unittest.TestCase):

    def test_action_selection(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        other_classifier = CIClassifier(previous_results, last_executions, average_duration, "Bfons", 42)
        match_set = [classifier, other_classifier]
        action_selector = ActionSelection(["Alfons", "Bfons"], 0)
        action_set = action_selector.get_action_set(match_set, "Alfons")
        assert len(action_set) == 1
        assert action_set[0] == classifier

    def test_select_action(self):
        prediction_array = {}
        prediction_array["Alfons"] = 100
        prediction_array["Bfons"] = 10
        action_selector = ActionSelection(["Alfons", "Bfons"], 0)
        action = action_selector.select_action(prediction_array)
        assert action == "Alfons"

    def test_calc_prediction_array(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier1 = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier1.fitness = 10.0
        classifier1.prediction = 5.0
        classifier2 = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        classifier2.fitness = 3.0
        classifier2.prediction = 4.0
        classifier3 = CIClassifier(previous_results, last_executions, average_duration, "Bfons", 42)
        classifier3.fitness = 7.0
        classifier3.prediction = 2.0
        match_set = [classifier1, classifier2, classifier3]
        action_selector = ActionSelection(["Alfons", "Bfons"], 0)
        prediction_array = action_selector.get_prediction_array(match_set)
        assert prediction_array["Alfons"] == (62.0 / 13)
        assert prediction_array["Bfons"] == 14 / 7.0


if "__main__" == __name__:

    unittest.main()

