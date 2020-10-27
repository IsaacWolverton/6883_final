# file with all the classes for different hyperparameter tuning
# TODO: JULIA - implement EvolutionTuning
#       ANISHA - implement BayesianOptimization

import random

# is this for individual parameters?
class HyperParamTuning:
    
    def __init__(self, range_min, range_max):
        self.range_min = range_min
        # TODO: note whether max is inclusive
        self.range_max = range_max
        self.current = 0
    def next_value(self):
        raise NotImplementedError("Implement this function when you extend the class")
    

class RandomSearch(HyperParamTuning):

    def next_value(self):
        self.current = random.uniform(self.range_min, self.range_max)
        return self.current

class GridSearch(HyperParamTuning):
    def __init__(self, range_min, range_max, number_of_splits):
        super().__init__(range_min, range_max)
        step = (range_max - range_min)/number_of_splits
        self.values = iter([range_min + i * step for i in range(number_of_splits+1)])

    def next_value(self):
        # TODO: decide where to handle stop iteration error
        return next(self.values)

class EvolutionTuning(HyperParamTuning):
    def __init__(self, range_min, range_max, population_size):
        super().__init__(range_min, range_max)
        self.values = [random.uniform(self.range_min, self.range_max) for i in range(population_size)]

    '''
    Once we've evaluated all of population, top X% will crossover to make offspring, and some mutations
    are introduced.
    '''
    def next_value(self):
        raise NotImplementedError

        
class BayesianOptimization(HyperParamTuning):
    def next_value(self):
        raise NotImplementedError