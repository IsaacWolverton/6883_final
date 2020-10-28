from option_keyboard.gpe_gpi_experiments.hp_optimizers import HyperParamTuner
from option_keyboard import run_hyperparameter_search

def my_func(params):
    val = params[0]
    return (val**2)

if __name__ == "__main__":
    hyperParamTuner = HyperParamTuner(my_func)
    range_min = 0.7
    range_max = 1
    # print (hyperParamTuner.tuneWithRandomSearch(range_min, range_max, 10))
    # print (hyperParamTuner.tuneWithGridSearch(range_min, range_max, 10))
    # print (hyperParamTuner.tuneWithBayesianOptimization(range_min, range_max))

    test_dqn_model = HyperParamTuner(run_hyperparameter_search.evaluate_dqn)
    print (test_dqn_model.tuneWithRandomSearch(range_min, range_max, 10))
    print (test_dqn_model.tuneWithGridSearch(range_min, range_max, 10))
    print (test_dqn_model.tuneWithBayesianOptimization(range_min, range_max))
