from abc import abstractmethod
from classifier import CIClassifier
import random
import unittest


class Matching:

    def __init__(self, theta_mna, available_actions):
        self.theta_mna = theta_mna
        self.available_actions = available_actions

    @abstractmethod
    def get_match_set(self, population, sigma, timestamp):
        pass

    @abstractmethod
    def generate_covering_classifier(self, sigma, missing_actions, timestamp):
        pass


class CIMatching(Matching):
    MAX_PAST = 0.5
    DURATION = 0.5
    P_DONT_CARE = 0.33
    ID = 0

    def __init__(self, theta_mna, available_actions):
        #random.seed(42)
        self.theta_mna = theta_mna
        self.available_actions = available_actions
        self.ID = CIMatching.ID
        CIMatching.ID += 1

    def get_match_set(self, population, sigma, timestamp):
        '''
        Collects all classifiers matching the situation sigma. Generates new classifiers
        if there is not enough diversity in the action space of the match set.

        TODO: add deletion

        :param population: list of classifiers.
        :param sigma: situation to be matched.
        :param timestamp: current time (discrete).

        :return : list of matching classifiers.
        '''
        match_set = []
        counter = 0
        while len(match_set) == 0:
            match_set = list(filter(lambda x: x.matches(sigma), population))
            available_actions = list(set(map(lambda x: x.action, match_set)))
            if len(available_actions) < self.theta_mna:
                for _ in range(0, self.theta_mna - len(available_actions)):
                    missing_actions = list(filter(lambda x: x not in available_actions, self.available_actions))
                    # covering
                    new_classifier = self.generate_covering_classifier(sigma, missing_actions, timestamp)
                    available_actions.append(new_classifier.action)
                    population.append(new_classifier)
                    assert new_classifier.matches(sigma)
                # delete
                match_set = []
            counter += 1
            assert counter < 10
        return match_set

    def generate_covering_classifier(self, sigma, missing_actions, timestamp):
        '''
        Generates a new classifier for a random missing action. The new classifier
        matches the situation sigma.

        :param sigma: situation
        :param missing_actions: list containing the actions not found in match set
        :param time_stamp: Current time (discrete)

        :return : new classifier.
        '''
        action = missing_actions[int(random.random() * len(missing_actions))]
        duration = sigma["duration"]
        # generate random duration condition
        duration_border = random.random() * CIMatching.DURATION
        duration_condition = (duration - duration_border, duration + duration_border)
        previous_results = sigma["previous_results"]
        previous_results_conditions = []
        # generate ternary conditions for results
        for i in range(0, len(previous_results)):
            if random.random() <= CIMatching.P_DONT_CARE:
                previous_results_conditions.append("#")
            else:
                previous_results_conditions.append(previous_results[i])
        # generate conditions
        last_execution = sigma["last_execution"]
        bound_lower = last_execution - random.random() * CIMatching.MAX_PAST
        bound_upper = last_execution + random.random() * CIMatching.MAX_PAST
        last_executions_condition = (bound_lower, bound_upper)
        classifier = CIClassifier(previous_results_conditions, last_executions_condition, duration_condition, action, timestamp)
        return classifier


class Test_CIMatching(unittest.TestCase):

    def test_match(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        average_duration = (41, 42)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        other_classifier = CIClassifier(previous_results, last_executions, average_duration, "Bfons", 42)
        situation = {}
        situation["duration"] = 43
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 2
        population = [classifier, other_classifier]
        matcher = CIMatching(1, ["Alfons", "Bfons"])
        match_set = matcher.get_match_set(population, situation, 1)
        assert len(match_set) == 1
        assert match_set[0] == classifier

    def test_covering(self):
        situation = {}
        situation["duration"] = 43
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 2
        matcher = CIMatching(1, ["Alfons", "Bfons"])
        classifier = matcher.generate_covering_classifier(situation, ["alfons"], 1)
        assert classifier.matches(situation) == True
        assert classifier.action == "alfons"
        assert classifier.timestamp == 1

    def test_match_with_covering(self):
        average_duration = (42, 45)
        previous_results = ["#", True, False]
        last_executions = (0, 3)
        classifier = CIClassifier(previous_results, last_executions, average_duration, "Alfons", 42)
        population = [classifier]
        situation = {}
        situation["duration"] = 48
        situation["previous_results"] = [True, True, False]
        situation["last_execution"] = 2
        matcher = CIMatching(2, ["alfons", "bfons"])
        matching = matcher.get_match_set(population, situation, 1)
        actions = set(map(lambda x: x.action, matching))
        assert len(population) == 3
        assert len(matching) == 2
        assert len(actions) == 2


if "__main__" == __name__:

    unittest.main()

