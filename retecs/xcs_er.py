import unittest
import copy
from classifier import CIClassifier
from matching import CIMatching
from action_selection import ActionSelection
from reinforcement import Reinforcement
from genetic_algorithm import CIGeneticAlgorithm
import random
import pickle
import numpy as np


class XCSExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount
        self.last_ci = 0

    def remember(self, experience):
        self.memory.append(experience)

    def get_batch(self, batch_size=10, last_ci=0):
        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]
        useable_memory = list(filter(lambda x: x[3] < last_ci, self.memory))
        if batch_size < len(useable_memory):
            timerank = range(1, len(useable_memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
            batch_idx = np.random.choice(range(len(useable_memory)), replace=False, size=batch_size, p=p)
            batch = [useable_memory[idx] for idx in batch_idx]
        else:
            batch = useable_memory

        return batch

    def get_get_exp_of_CI_cyle(self, cycle_id):
        # state, action, reward, cycle_id
        return list(filter(lambda x: x[3] == cycle_id, self.memory))


class XCS_ER:

    GAMMA = 0.71

    def __init__(self, max_population_size, possible_actions=[], histlen=42):
        self.name = "XCS_ER"
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
        self.rewards = None
        self.p_explore = 0.25
        self.train_mode = True
        #################################
        # dumb idea that will never work
        #################################
        self.experience_length = 12000
        self.experience_batch_size = 2000
        self.experience = XCSExperienceReplay(max_memory=self.experience_length)
        self.ci_cycle = 0

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
        self.action_history.append((state, action))
        return action

    def reward(self, new_rewards):
        try:
            x = float(new_rewards)
            new_rewards = [x] * len(self.action_history)
        except Exception as _:
            if len(new_rewards) < len(self.action_history):
                raise Exception('Too few rewards')
        for i in range(0, len(new_rewards)):
            reward = new_rewards[i]
            state, action = self.action_history[i]
            self.experience.remember((state, action, reward, self.ci_cycle))
        self.action_history = []
        self.ci_cycle += 1
        if self.ci_cycle == 2 or self.ci_cycle % 3 == 0:
            print("start ER")
            self.learn_from_experience()
            print("finish ER")
        print("finished CI cyle " + str(self.ci_cycle - 1))

    def get_average_prediction(self, cycle_id, on_policy=False):
        next_experiences = self.experience.get_get_exp_of_CI_cyle(cycle_id + 1)
        if next_experiences is None:
            return None
        prediction_sum = 0
        for old_experience in next_experiences:
            state, _, _, _ = old_experience
            theta_mna = len(self.possible_actions)
            matcher = CIMatching(theta_mna, self.possible_actions)
            match_set = matcher.get_match_set(self.population, state, self.time_stamp)
            action_selector = ActionSelection(self.possible_actions, 0)
            prediction_array = action_selector.get_prediction_array(match_set)
            action = action_selector.select_action(prediction_array, self.train_mode)
            if on_policy:
                prediction_sum += prediction_array[action]
            else:
                prediction_sum += max(prediction_array.keys(), key=(lambda k: prediction_array[k]))
        return prediction_sum / len(next_experiences)

    def learn_from_experience(self):
        experiences = self.experience.get_batch(self.experience_batch_size, self.ci_cycle - 1)
        states, actions, rewards, ci_cyles = zip(*experiences)
        cycles_of_batch = set(ci_cyles)
        prediction_vals = {}
        for cycle_id in cycles_of_batch:
            prediction_vals[cycle_id] = self.get_average_prediction(cycle_id, False)
        print("retrieved prediction approx.")
        for i in range(0, len(rewards)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            cycle = ci_cyles[i]
            if prediction_vals[cycle] is not None:
                discounted_reward = reward + XCS_ER.GAMMA * prediction_vals[cycle]
                # match set
                theta_mna = len(self.possible_actions)
                # use covering?
                # len(self.possible_actions)
                matcher = CIMatching(theta_mna, self.possible_actions)
                match_set = matcher.get_match_set(self.population, state, self.time_stamp)
                # action_set
                action_selector = ActionSelection(self.possible_actions, self.p_explore)
                action_set = action_selector.get_action_set(match_set, action)
                if len(action_set) > 0:
                    # update classifiers
                    self.reinforce.reinforce(action_set, discounted_reward)
                    self.ga.perform_iteration(action_set, state, self.population, self.time_stamp)
                    self.time_stamp += 1
            if i % 10 == 0:
                print("finished " + str(i / len(rewards)) + " percent of ER")
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
