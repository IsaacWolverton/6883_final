# file with all the classes for different hyperparameter tuning
import random
from skopt import gp_minimize

'''
This class is for tuning hyperparameters. It takes in an evaluate model which should be a method
that takes in a list of hyperparameters and outputs the objective value. Each method will run
tuning using the corresponding type and return the optimal hyperparameters in a list. Note that
each method will thus run evaluate_model multiple times in order to evaluate several different
hyperparameter possibilities. 
'''
class HyperParamTuning:
    
    def __init__(self, evaluate_model):
        self.evaluate_model = evaluate_model    

    '''
    This performs random search using the evaluate model and returns a list of the params that 
    evaluate to the highest objective value.
    range_min (float): lower, inclusive bound of the range to random search
    range_max (float): upper, inclusive bound of the range to random search
    iterations (integer): number of times to evaluate on the random parameters 
    '''
    def tuneWithRandomSearch(self, range_min, range_max, iterations):
        best_objective_val = None
        best_params = None
        for i in range(iterations):
            params = [random.uniform(range_min, range_max)]
            current_objective_val = self.evaluate_model([params])
            if not best_objective_val or current_objective_val > best_objective_val:
                best_params = params
                best_objective_val = current_objective_val  
        return best_params

    '''
    This performs grid search using the evaluate model and returns a list of the params that 
    evaluate to the highest objective value.
    range_min (float): lower, inclusive bound of the range to random search
    range_max (float): upper, inclusive bound of the range to random search
    number_of_splits (integer): number of times to evaluate on the parameters, in 
                                (range_max-range_min)/number_of_splits increments
    '''
    def tuneWithGridSearch(self, range_min, range_max, number_of_splits):
        step = (range_max - range_min)/number_of_splits
        best_objective_val = None
        best_params = None
        for i in range(number_of_splits+1):
            params = [range_min + i * step]
            current_objective_val = self.evaluate_model([params])
            if not best_objective_val or current_objective_val > best_objective_val:
                best_params = params
                best_objective_val = current_objective_val  
        return best_params

    '''
    This performs bayesian optimization using the evaluate model and returns a list of the params that 
    evaluate to the highest objective value.
    range_min (float): lower, exclusive bound of the range to random search
    range_max (float): upper, exclusive bound of the range to random search
    '''
    def tuneWithBayesianOptimization(self, range_min, range_max):
        return gp_minimize(self.evaluate_model, [float(range_min), float(range_max)])