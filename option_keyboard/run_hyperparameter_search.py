# functions that call experiment.py, and run an experiment with
# a regressed_agent or a dqn_agent and an environment. 

# ============================================================================

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

from option_keyboard import configs
from option_keyboard import dqn_agent
from option_keyboard import environment_wrappers
from option_keyboard import experiment
from option_keyboard import scavenger

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of training episodes.")
flags.DEFINE_integer("report_every", 200,
                     "Frequency at which metrics are reported.")
flags.DEFINE_string("output_path", None, "Path to write out training curves.")

# ===========================================================================

# in one lifetime, create an environment, pass in next_value as its additional_discount, run the experiment
def run_hyperparam_regressed(discount_list):
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
  additional_discount = 0.9
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
      num_episodes=FLAGS.num_episodes,
      report_every=FLAGS.report_every,
      num_eval_reps=20)
  if FLAGS.output_path:
    experiment.write_returns_to_file(FLAGS.output_path, ema_returns)


def run_hyperparam_dqn(discount_list):
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
      num_episodes=FLAGS.num_episodes,
      report_every=FLAGS.report_every)
  if FLAGS.output_path:
    experiment.write_returns_to_file(FLAGS.output_path, ema_returns)
  
  return ema_returns[-1]

