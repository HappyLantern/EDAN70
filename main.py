from pysc2.env import sc2_env
from pysc2.env import available_actions_printer
from pysc2 import maps
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.lib import point_flag
from pysc2.agents import scripted_agent
from pysc2.agents import base_agent
#from .first_agent import Agent

from absl import flags
from absl import app

import importlib
import sys
import time
import threading

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 1000, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "first_agent.TestAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "rand   om", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")

def run_loop(agents, env, max_agent_steps, max_episodes):

    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)

    agent = agents[0]

    history = []
    try:
       for ep in range(1, max_episodes):
        # while True:
            total_episodes += 1
            done = False
            total_reward = 0
            timesteps = env.reset()
            #agent.reset()

            while not done:
                #total_frames += 1
                # Ask agent for actions
                actions = [agent.step(timestep)
                       for agent, timestep in zip(agents, timesteps)]

                # Apply actions to the environment.
                timesteps_new = env.step(actions)
                # Record the <s, a, r, s'> for training
                agent.record_step(timesteps, actions, timesteps_new)
                timesteps = timesteps_new

                # Train the network after each episode
                agent.update()
    except KeyboardInterrupt:
        pass
    finally:
        print("Looks good")




def run_thread(agent_classes, players, map_name, visualize):
    with sc2_env.SC2Env(
        map_name=FLAGS.map,
        players=players,
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=FLAGS.feature_screen_size,
            feature_minimap=FLAGS.feature_minimap_size,
            rgb_screen=FLAGS.rgb_screen_size,
            rgb_minimap=FLAGS.rgb_minimap_size,
            action_space=FLAGS.action_space,
            use_feature_units=FLAGS.use_feature_units),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=FLAGS.game_steps_per_episode,
        disable_fog=FLAGS.disable_fog,
        visualize=False) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agents = [agent_cls() for agent_cls in agent_classes]
        run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)


def main(unused_argv):

    agent_classes = []
    players = []

    # Add agent class/module and players
    agent_type, agent_name = FLAGS.agent.rsplit(".", 1) # split the agent and agent name
    agent_class = getattr(importlib.import_module(agent_type), agent_name) # RandomAgent object
    agent_classes.append(agent_class)

    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                               FLAGS.agent_name or agent_name))

    run_thread(agent_classes, players, FLAGS.map, FLAGS.render)




#
if __name__ == '__main__':
    app.run(main)
