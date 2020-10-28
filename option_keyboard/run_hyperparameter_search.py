# functions that call experiment.py, and run an experiment with
# a regressed_agent or a dqn_agent and an environment. 

# ============================================================================

# from absl import app
# from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from option_keyboard import configs
from option_keyboard import dqn_agent
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import scavenger
from option_keyboard import smart_module
from option_keyboard.gpe_gpi_experiments import regressed_agent

# FLAGS = flags.FLAGS
# flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
# flags.DEFINE_integer("report_every", 200,
#                      "Frequency at which metrics are reported.")
# flags.DEFINE_string("output_path", None, "Path to write out training curves.")
# flags.DEFINE_string("keyboard_path", None, "Path to keyboard model.")
num_episodes = 10000
report_every = 200
output_path = None
keyboard_path = None

# ===========================================================================

# in one lifetime, create an environment, pass in next_value as its additional_discount, run the experiment

# TODO if we want to pass in a custom discount to the keyboard, we need to:
    # make a new version of keyboard_utils where additional_discount is not .9, but passed in
    # make a new version of train_keyboard where the create_and_train_keyboard function takes in a speicifc discount 
    # figure out the parallelization stuff re: naming the keyboard file

def evaluate_regressed(discount_list):
  """
  Function that takes in a discount value and returns the objective value for regressed
  
  params:
    discount_list = list of length 1 with the discount
  returns:
    objective value at end of training 
  """
  keyboard = smart_module.SmartModuleImport(hub.Module(FLAGS.keyboard_path))

  # Create the task environment.
  base_env_config = configs.get_fig4_task_config()
  base_env = scavenger.Scavenger(**base_env_config)
  base_env = environment_wrappers.EnvironmentWithLogging(base_env)

  # Wrap the task environment with the keyboard.
  additional_discount = discount_list[0]
  env = environment_wrappers.EnvironmentWithKeyboardDirect(
      env=base_env,
      keyboard=keyboard,
      keyboard_ckpt_path=None,
      additional_discount=additional_discount,
      call_and_return=False)

  # Create the player agent.
  agent = regressed_agent.Agent(
      batch_size=10,
      optimizer_name="AdamOptimizer",
      optimizer_kwargs=dict(learning_rate=3e-2,),
      init_w=np.random.normal(size=keyboard.num_cumulants) * 0.1,
  )

  _, ema_returns = experiment.run(
      env,
      agent,
      num_episodes=num_episodes,
      report_every=report_every,
      num_eval_reps=20)
  if output_path:
    experiment.write_returns_to_file(output_path, ema_returns)


def evaluate_dqn(discount_list):
  """
  Function that takes in a discount value and returns the objective value for DQN

  params:
    discount_list = list of length 1 with the discount
  returns:
    objective value at end of training
  """

  env_config = configs.get_task_config()
  env = scavenger.Scavenger(**env_config)
  env = environment_wrappers.EnvironmentWithLogging(env)

  # Create the dqn agent.
  agent = dqn_agent.Agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      network_kwargs=dict(
          output_sizes=(64, 128),
          activate_final=True,
      ),
      epsilon=0.1,
      # additional discount is our manually passed in value
      additional_discount = discount_list[0],
      batch_size=10,
      optimizer_name="AdamOptimizer",
      optimizer_kwargs=dict(learning_rate=3e-4,))

  _, ema_returns = experiment.run(
      env,
      agent,
      num_episodes=num_episodes,
      report_every=report_every)
  if output_path:
    experiment.write_returns_to_file(output_path, ema_returns)
  
  # pull out the evaluation reward from the final episode
  final_eval_reward = ema_returns[-1].get("eval")[0]

  # return the negative of that value to the minimization function(s)
  return final_eval_reward * -1


def main():

  dis_list = [.9]
  obj_val = evaluate_dqn(dis_list)

if __name__ == "__main__":
  tf.disable_v2_behavior()
  main()
