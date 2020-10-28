from hp_optimizers import HyperParamTuner

def my_func(params):
    val = params[0]
    return (val**2)

if __name__ == "__main__":
    hyperParamTuner = HyperParamTuner(my_func)
    range_min = -1
    range_max = 1
    print (hyperParamTuner.tuneWithRandomSearch(range_min, range_max, 10))
    print (hyperParamTuner.tuneWithGridSearch(range_min, range_max, 10))
    print (hyperParamTuner.tuneWithBayesianOptimization(range_min, range_max))