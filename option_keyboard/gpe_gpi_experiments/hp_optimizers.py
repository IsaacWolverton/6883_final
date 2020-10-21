# file with all the classes for different hyperparameter tuning
# TODO: JULIA - implement EvolutionTuning
#       ANISHA - implement BayesianOptimization

import random

class HyperParamTuning:
    
    def __init__(self, range_min, range_max):
        self.range_min = range_min
        self.range_max = range_max

    def next_value(self):
        raise NotImplementedError("Implement this function when you extend the class")
    

class RandomSearch(HyperParamTuning):

    def next_value(self):
        return random.uniform(self.range_min, self.range_max)


class EvolutionTuning(HyperParamTuning):
    def next_value(self):
        raise NotImplementedError

        
class BayesianOptimization(HyperParamTuning):
    def next_value(self):
        raise NotImplementedError
