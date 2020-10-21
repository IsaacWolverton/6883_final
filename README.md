# 6883_final
6.883 Fast RL with self-tuning hyperparameters

### Setup
Use below command to setup virtualenv, only specify --python if you need to
`virtualenv --python=/usr/bin/python3.7 venv`

Activate this virtualenv
`source venv/bin/activate`

Install the python libraries
`pip install -r option_keyboard/requirements.txt`

### Train the DQN baseline
`python -m option_keyboard.run_dqn`

### Train the Option Keyboard and agent
`python -m option_keyboard.run_ok`
