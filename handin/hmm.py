# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier, CodeBook
from collections import Counter
from corpus import Document


class SingleSentence(Document):
    def features(self):
        return self.data


class HMM(Classifier):
    """A Hidden Markov Model classifier."""
    def __init__(self):
        super(HMM, self).__init__()
        self.initial_probabilities = []
        self.transition_probabilities = []
        self.emission_probabilities = []
        self.final_probabilities = []

        self.state_code_book = CodeBook("")
        self.observation_code_book = CodeBook("")
        self.observation_counter = Counter()

    def get_model(self):
        return None

    def set_model(self, model):
        pass
    model = property(get_model, set_model)

    def train_ice_cream(self, instances,
                        initial_probabilities=None,
                        transition_probabilities=None,
                        emission_probabilities=None,
                        states=None,
                        vocabulary=None):
        """traning for ice cream test"""
        self.initial_probabilities = initial_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.state_code_book = CodeBook(states)
        self.observation_code_book = CodeBook(vocabulary)

    def train(self, instances):
        """Supervised training for viterbi classifier"""
        observation_counter_oj = Counter()
        observation_counter_j = Counter()
        transition_counter = Counter()
        initial_counter = Counter()

        # for words and their POS-tags in each instance, count the appearance by counting their coded indices
        for instance in instances:
            for index in range(len(instance.features())):
                self.state_code_book.add(instance.label[index])
                self.observation_code_book.add(instance.data[index])
                observation_counter_oj[(self.observation_code_book[instance.data[index]],
                                        self.state_code_book[instance.label[index]])] += 1
                observation_counter_j[self.state_code_book[instance.label[index]]] += 1
                if index != 0:
                    transition_counter[(self.state_code_book[instance.label[index - 1]],
                                        self.state_code_book[instance.label[index]])] += 1
                else:
                    initial_counter[self.state_code_book[instance.label[index]]] += 1

        #initial prob list and give the combinations that not appearing in the training data a smoothing probability
        smooth_value = 1.0/sum(observation_counter_j.values())
        smooth_value_initial = 1.0/(sum(initial_counter.values())+2)
        self.initial_probabilities = [smooth_value_initial]*len(self.state_code_book)
        self.transition_probabilities = [[smooth_value for j in range(len(self.state_code_book))]
                                         for i in range(len(self.state_code_book))]
        self.emission_probabilities = [[smooth_value for o in range(len(self.observation_code_book))]
                                       for j in range(len(self.state_code_book))]

        # compute each probability and smoothing
        for (i, j) in transition_counter:
            self.transition_probabilities[i][j] = \
                (transition_counter[(i, j)]+1.0)/(observation_counter_j[i]+2)

        for (o, j) in observation_counter_oj:
            self.emission_probabilities[j][o] = \
                (observation_counter_oj[o, j]+1.0)/(observation_counter_j[j]+2)

        for j in initial_counter:
            self.initial_probabilities[j] = (initial_counter[j]+1.0)/(sum(initial_counter.values())+2)

        #store the observation_counter_j for smoothing when classifying
        self.observation_counter = observation_counter_j

    def classify(self, instance):
        """use the viterbi algorithm to classify"""
        length_of_input = len(instance.data)
        number_of_states = len(self.state_code_book)
        backtrace_path = [0]*length_of_input
        back_pointer = [[0 for i in range(number_of_states)] for j in range(length_of_input)]
        viterbi = [[0 for i in range(number_of_states)] for j in range(length_of_input)]

        # compute by using viterbi, and deal with unknown words
        for t in range(length_of_input):
            if instance.data[t] not in self.observation_code_book:
                self.observation_code_book.add(instance.data[t])
                for j in range(number_of_states):
                    self.emission_probabilities[j].append(1.0/self.observation_counter[j]+2)

            for j in range(number_of_states):
                if t == 0:
                    viterbi[t][j] = self.initial_probabilities[j] * \
                        self.emission_probabilities[j][self.observation_code_book[instance.data[0]]]
                else:
                    temp_list = []
                    for i in range(number_of_states):
                        temp_list.append(viterbi[t-1][i]*self.transition_probabilities[i][j] *
                                         self.emission_probabilities[j][self.observation_code_book[instance.data[t]]])
                    viterbi[t][j] = max(temp_list)
                    back_pointer[t][j] = temp_list.index(viterbi[t][j])

        final_state = viterbi[length_of_input-1].index(max(viterbi[length_of_input-1]))

        # get the back trace
        backtrace_path[length_of_input-1] = final_state
        for t in range(length_of_input-1, 0, -1):
            backtrace_path[t-1] = back_pointer[t][backtrace_path[t]]

        return [self.state_code_book.name(i) for i in backtrace_path]

    def likelihood(self, instance):
        """Compute the likelihood: P(O)"""
        number_of_states = len(self.state_code_book)
        length_of_input = len(instance.data)
        likelihood = .0
        alphas = self.compute_alpha(instance)
        for i in range(number_of_states):
            likelihood += alphas[length_of_input-1][i]
        return likelihood

    def compute_alpha(self, instance):
        number_of_states = len(self.state_code_book)
        length_of_input = len(instance.data)
        alphas = [[.0 for i in range(number_of_states)] for j in range(length_of_input)]

        for t in range(length_of_input):
            if instance.data[t] not in self.observation_code_book:
                self.observation_code_book.add(instance.data[t])
                for j in range(number_of_states):
                    self.emission_probabilities[j].append(1.0/self.observation_counter[j]+2)
            for j in range(number_of_states):
                if t == 0:
                    alphas[t][j] = (self.initial_probabilities[j] + .0) * \
                        self.emission_probabilities[j][self.observation_code_book[instance.data[0]]]
                else:
                    for i in range(number_of_states):
                        alphas[t][j] += (alphas[t-1][i]*self.transition_probabilities[i][j] + .0) * \
                            self.emission_probabilities[j][self.observation_code_book[instance.data[t]]]
        return alphas


class UnsupervisedHMM(HMM):

    def compute_alpha(self, instance):
        number_of_states = len(self.state_code_book)
        length_of_input = len(instance.data)
        alphas = [[.0 for i in range(number_of_states)] for j in range(length_of_input)]

        for t in range(length_of_input):

            for j in range(number_of_states):
                if t == 0:
                    alphas[t][j] = (self.initial_probabilities[j] + .0) * \
                        self.emission_probabilities[j][self.observation_code_book[instance.data[0]]]
                else:
                    for i in range(number_of_states):
                        alphas[t][j] += (alphas[t-1][i]*self.transition_probabilities[i][j] + .0) * \
                            self.emission_probabilities[j][self.observation_code_book[instance.data[t]]]
        return alphas

    def compute_beta(self, instance):
        number_of_states = len(self.state_code_book)
        length_of_input = len(instance.data)
        betas = [[.0 for i in range(number_of_states)] for j in range(length_of_input)]

        for t in range(length_of_input - 1, -1, -1):
            for i in range(number_of_states):
                if t == length_of_input - 1:
                    betas[length_of_input - 1][i] = self.final_probabilities[i]
                for j in range(number_of_states):
                    betas[t][i] += (betas[t+1][j]*self.transition_probabilities[i][j] + .0) * \
                        self.emission_probabilities[j][self.observation_code_book[instance.data[t+1]]]
        return betas

    def compute_gamma(self, alphas, betas, instance):
        number_of_states = len(self.state_code_book)
        gammas = [[.0 for i in range(number_of_states)] for t in range(len(instance.data)) ]

        for t in range(len(instance.data)):
            for j in range(number_of_states):
                gammas[t][j] = alphas[t][j] * betas[t][j]
        return gammas

    def compute_xi(self, alphas, betas, instance):
        number_of_states = len(self.state_code_book)
        xi = [[[.0 for i in range(number_of_states)] for j in range(number_of_states)] for t in range(len(instance.data) - 1)]

        for t in range(len(instance.data) - 1):
            for i in range(number_of_states):
                for j in range(number_of_states):
                    xi[t][i][j] = alphas[t][j] * self.transition_probabilities[i][j] * \
                        self.emission_probabilities[j][self.observation_code_book[instance.data[t]]] * betas[t+1][j]
        return xi

    def train(self, instances, iter_number=20):
        single_sequence = []
        for instance in instances:
            single_sequence.extend(instance.data)

        single_sequence = SingleSentence(single_sequence)

        # initialization
        self.state_code_book = CodeBook([i for i in range(10)])
        self.observation_code_book = CodeBook(single_sequence.data)
        self.final_probabilities = [1] * len(self.state_code_book)
        self.initial_probabilities = [0.1] * len(self.state_code_book)
        self.transition_probabilities = [[1.0/100 for j in range(len(self.state_code_book))]
                                         for i in range(len(self.state_code_book))]
        self.emission_probabilities = [[1.0/10 for o in range(len(self.observation_code_book.names))]
                                       for j in range(len(self.state_code_book))]

        alphas = self.compute_alpha(single_sequence)
        betas = self.compute_beta(single_sequence)

        for count in range(iter_number):
            # E-step
            gamma = self.compute_gamma(alphas, betas, single_sequence)
            xi = self.compute_xi(alphas, betas, single_sequence)

            # M-step
            for i in range(len(self.state_code_book)):
                for j in range(len(self.state_code_book)):
                    temp_up = sum(xi[t][i][j] for t in range(len(self.observation_code_book.names) - 1))
                    temp_down = .0
                    for t in range(len(self.observation_code_book.names) - 1):
                        for j2 in range(len(self.state_code_book)):
                            temp_down += xi[t][i][j2]
                    self.transition_probabilities[i][j] = temp_up/temp_down

            for j in range(len(self.state_code_book)):
                for o in self.observation_code_book.names.itervalues():
                    temp_up = .0
                    temp_down = .0
                    for t, word in enumerate(single_sequence.data):
                        if word == o:
                            temp_up += gamma[t][j]
                        temp_down += gamma[t][j]
                    self.emission_probabilities[j][self.observation_code_book[o]] = temp_up/temp_down


















