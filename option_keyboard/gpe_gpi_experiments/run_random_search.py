from option_keyboard.gpe_gpi_experiments.hp_optimizers import HyperParamTuner
from option_keyboard.run_hyperparameter_search import evaluate_dqn

if __name__ == "__main__":
    hyperParamTuner = HyperParamTuner(evaluate_dqn)
    range_min = .7
    range_max = 1

    hyperParamTuner.tuneWithRandomSearch(range_min, range_max, 100)