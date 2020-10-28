from option_keyboard.gpe_gpi_experiments.hp_optimizers import HyperParamTuner
from option_keyboard.run_hyperparameter_search import evaluate_dqn

def my_func(params):
    val = params[0]
    return (val**2)

if __name__ == "__main__":
    hyperParamTuner = HyperParamTuner(my_func)
    range_min = 0
    range_max = 1

    print (hyperParamTuner.tuneWithRandomSearch(range_min, range_max, 30))
    print (hyperParamTuner.tuneWithGridSearch(range_min, range_max, 30))
    print (hyperParamTuner.tuneWithBayesianOptimization(range_min, range_max))