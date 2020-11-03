# 6883_final
6.883 Fast RL with self-tuning hyperparameters, this project builds off of the work done by the DeepMind team [here](https://github.com/deepmind/deepmind-research/tree/master/option_keyboard).

Below is a table comparing the three hyperparameter tuning algorithms. Each algorithm was run for 100 agent lifetimes.

Method | Discount | Cumulative Reward
------------ | ------------- | -------------
Grid Search | 0.852 | 6.8885
Random Search | 0.912 | 6.9245
Bayesian Search | 0.885 | 7.0609

### Setup
Use below command to setup virtualenv, only specify --python if you need to:
```
virtualenv --python=/usr/bin/python3.7 venv
```

Activate this virtualenv:
```
source venv/bin/activate
```

Install the python libraries:
```
pip install -r option_keyboard/requirements.txt
```

### Train the DQN baseline
```
python -m option_keyboard.run_dqn
```

### Train the self-tuning DQN
```
python -m option_keyboard.run_meta_dqn
```

### Train the Option Keyboard and agent
```
python -m option_keyboard.run_ok
```

### Our Files (for 6.883, not from Deepmind)
- meta_dqn_agent.py
- run_meta_dqn.py
#### In the parameter-tuning branch:

- run_hyperparameter_search.py
- hp_optimizers.py
- run_bayesian.py
- run_grid_search.py
- run_random_search.py
- test.py

