"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf
import functools

# local submodule
import agents.networks.network as nets

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS

# action space
attack_screen = functools.partial(FUNCTIONS.Attack_screen, "now")
move_screen   = functools.partial(FUNCTIONS.Move_screen, "now")
select_point  = functools.partial(FUNCTIONS.select_point, "toggle")
action_space  = [attack_screen, move_screen, select_point]
NUM_ACTIONS   = len(action_space)


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False)

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class DQNAgent(base_agent.BaseAgent):
    """A DQN that receives screen observations and selects units, moves and/or attacks."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 discount_factor=FLAGS.discount_factor,
                 epsilon_max=FLAGS.epsilon_max,
                 epsilon_min=FLAGS.epsilon_min,
                 epsilon_decay_steps=FLAGS.epsilon_decay_steps,
                 train_frequency=FLAGS.train_frequency,
                 target_update_frequency=FLAGS.target_update_frequency,
                 max_memory=FLAGS.max_memory,
                 batch_size=FLAGS.batch_size,
                 training=FLAGS.training,
                 indicate_nonrandom_action=FLAGS.indicate_nonrandom_action,
                 save_dir="./checkpoints/",
                 ckpt_name="DQNAgent",
                 summary_path="./tensorboard/deepq"):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNAgent, self).__init__()

        # saving and summary writing
        if FLAGS.save_dir:
            save_dir = FLAGS.save_dir
        if FLAGS.ckpt_name:
            ckpt_name = FLAGS.ckpt_name
        if FLAGS.summary_path:
            summary_path = FLAGS.summary_path

        # neural net hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # agent hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

        # other parameters
        self.training = training
        self.indicate_nonrandom_action = indicate_nonrandom_action

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        tf.reset_default_graph()
        self.network = nets.Network(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=summary_path)

        if self.training:
            self.target_net = nets.Network(
                spatial_dimensions=feature_screen_size,
                learning_rate=self.learning_rate,
                name="DQNTarget")

            # initialize Experience Replay memory buffer
            self.memory = Memory(max_memory)
            self.batch_size = batch_size

        print("Done.")

        self.last_state = None
        self.last_action = None

        # initialize session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
            if self.training:
                self._update_target_network()
        else:
            self._tf_init_op()

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.reward = 0

        if self.training:
            self.last_state = None
            self.last_action = None

            global_episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", global_episode + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        state = obs.observation.feature_screen
        if self.training:
            # predict an action to take and take it
            action, action_index = self._epsilon_greedy_action_selection(state)

            # update online DQN
            if (self.steps % self.train_frequency == 0 and
                    len(self.memory) > self.batch_size):
                self._train_network()

            # update network used to estimate TD targets
            if self.steps % self.target_update_frequency == 0:
                self._update_target_network()
                print("Target network updated.")

            # add experience to memory
            if self.last_state is not None:
                self.memory.add(
                    (self.last_state,
                     self.last_action,
                     obs.reward,
                     state))

            self.last_state = state
            self.last_action = action_index

            
        else:
            action, _ = self._epsilon_greedy_action_selection(
                state,
                self.epsilon_min)

        action_name = action.function.name
        if action_name == 'select_point':
            action_id = 2
        elif action_name == 'Attack_screen':
            action_id = 12
        else:
            action_id = 331
        return action if action_id in obs.observation.available_actions else FUNCTIONS.no_op()

    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # increment global training episode
        self.network.increment_global_episode_op(self.sess)

        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        states, actions, targets = self._get_batch()
        self.network.write_summary(
            self.sess, states, actions, targets, self.reward)
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQNTarget")

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)

    def _epsilon_greedy_action_selection(self, state, epsilon=None):
        """Choose action from state with epsilon greedy strategy."""
        step = self.network.global_step.eval(session=self.sess)

        if epsilon is None:
            epsilon = max(
                self.epsilon_min,
                (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) *
                                     step / self.epsilon_decay_steps)))

        if epsilon > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])
            action_index = np.random.randint(0, len(action_space))
            action = action_space[action_index]
            flat_index = np.ravel_multi_index((x, y, action_index), (84, 84, NUM_ACTIONS))
            return action((x, y)), flat_index

        else:
            inputs = np.expand_dims(state, 0)

            q_values = self.sess.run(
                self.network.output,
                feed_dict={self.network.inputs: inputs})

            max_index = np.argmax(q_values)
            x, y, action_index = np.unravel_index(max_index, (84, 84, NUM_ACTIONS))
            #print(action_index)
            action = action_space[action_index]
            return action((x, y)), max_index

    def _train_network(self):
        states, actions, targets = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, targets)

    def _get_batch(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # one-hot encode actions
        actions = np.eye(np.prod((84, 84, NUM_ACTIONS)))[actions]

        # get targets
        next_outputs = self.sess.run(
            self.target_net.output,
            feed_dict={self.target_net.inputs: next_states})

        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets
