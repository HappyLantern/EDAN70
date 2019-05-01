import tensorflow as tf
from tensorflow.contrib import layers
from pysc2.lib import actions
from pysc2.lib import features
from expreplay import ExpReplay
import numpy as np
from utils import preprocess_minimap, preprocess_screen

NUM_SCREEN_FEATURES = len(features.SCREEN_FEATURES)
NUM_MINIMAP_FEATURES = len(features.MINIMAP_FEATURES)
SCREEN_SIZE = (64, 64)
MINIMAP_SIZE = (64, 64)
NUM_ACTIONS = len(actions.FUNCTIONS)


class TestAgent():

    """ Initialize agent """
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.tf_session = None

        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.replay = ExpReplay(10000)
        print("\n" + "-+"*40, "\nCreate network here\n" + "-+"*40, "\n")

        # Placeholders for observation
        self.minimap = tf.placeholder(shape=[None, NUM_MINIMAP_FEATURES, *MINIMAP_SIZE],
                                      dtype=tf.int32)
        self.screen = tf.placeholder(shape=[None, NUM_SCREEN_FEATURES, *SCREEN_SIZE],
                                     dtype=tf.int32)
        self.non_spatial = tf.placeholder(shape=[None, NUM_ACTIONS],
                                          dtype=tf.float32)

        # Preprocess for net feed
        self.screen_processed = preprocess_screen(self.screen)
        self.minimap_processed = preprocess_minimap(self.minimap)

        # Minimap conv layers
        self.mconv1 = layers.conv2d(self.minimap_processed,
                                    num_outputs=16,
                                    kernel_size=5,
                                    stride=1,
                                    scope='mconv1')
        self.mconv2 = layers.conv2d(self.mconv1,
                                    num_outputs=32,
                                    kernel_size=3,
                                    stride=1,
                                    scope='mconv2')
        # Screen conv layers
        self.sconv1 = layers.conv2d(self.screen_processed,
                                    num_outputs=16,
                                    kernel_size=5,
                                    stride=1,
                                    scope='sconv1')
        self.sconv2 = layers.conv2d(self.sconv1,
                                    num_outputs=32,
                                    kernel_size=3,
                                    stride=1,
                                    scope='sconv2')
        # Non spatial layer
        self.info_fc = layers.fully_connected(layers.flatten(self.non_spatial),
                                              num_outputs=256,
                                              activation_fn=tf.tanh,
                                              scope='info_fc')

        # Compute spatial actions
        self.feat_conv = tf.concat([self.mconv2, self.sconv2], axis=3) # Axis 3=channels differ in size
        self.spatial_action = layers.conv2d(self.feat_conv,
                                            num_outputs=1,
                                            kernel_size=1,
                                            stride=1,
                                            activation_fn=None,
                                            scope='spatial_action')
        self.spatial_action = tf.nn.softmax(
            layers.flatten(self.spatial_action))

        # Compute non spatial actions and value
        self.feat_fc = tf.concat(
            [layers.flatten(self.mconv2), layers.flatten(self.sconv2), self.info_fc], axis=1)
        self.feat_fc = layers.fully_connected(self.feat_fc,
                                              num_outputs=256,
                                              activation_fn=tf.nn.relu,
                                              scope='feat_fc')
        self.non_spatial_action = layers.fully_connected(self.feat_fc,
                                                         num_outputs=NUM_ACTIONS,
                                                         activation_fn=tf.nn.softmax,
                                                         scope='non_spatial_action')
        self.value = tf.reshape(layers.fully_connected(self.feat_fc,
                                                       num_outputs=1,
                                                       activation_fn=None,
                                                       scope='value'), [-1])

                                                       # t_vars = tf.trainable_variables()
                                                       # self.indices = tf.range(0, tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()]))
        # Train network

        # Set targets and masks
        self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
        self.spatial_action_selected = tf.placeholder(tf.float32, [None, SCREEN_SIZE[0]**2], name='spatial_action_selected')
        self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
        self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
        self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

        # Compute log probability
        spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
        spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
        non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
        valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
        valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
        non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
        non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

        # Policy loss and value loss
        action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
        advantage = tf.stop_gradient(self.value_target - self.value)
        policy_loss = - tf.reduce_mean(action_log_prob * advantage)
        value_loss = - tf.reduce_mean(self.value * advantage)
        loss = policy_loss + value_loss # Policy penalty

        # Build the optimizer
        opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
        grads = opt.compute_gradients(loss)
        self.train_op = opt.apply_gradients(grads)

    """ Choose actions based on observation (timestep) """
    def step(self, timestep):
        self.steps += 1
        self.reward += timestep.reward

        # screen_features
        observation = timestep.observation
        screen_features = observation.feature_screen
        screen_features = np.expand_dims(screen_features, axis=0)

        # minimap_features
        minimap_features = observation.feature_minimap
        minimap_features = np.expand_dims(minimap_features, axis=0)

        # actions
        available_actions = observation.available_actions
        action_mask = np.zeros(NUM_ACTIONS, dtype=np.int32)
        action_mask[available_actions] = 1
        action_mask = np.expand_dims(action_mask, axis=0)

        # get spatial and non spatial action
        feed_dict = {self.minimap: minimap_features,
                     self.screen: screen_features,
                     self.non_spatial: action_mask}
        non_spatial_action, spatial_action = self.tf_session.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed_dict)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel() # Same as flatten
        spatial_action = spatial_action.ravel() # Same as flatten
        valid_actions = timestep.observation.available_actions
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])] # Get action with highest prob
        target = np.argmax(spatial_action) # target (index) represents coordinate (size is 64*64)
        target = [int(target // 64), int(target % 64)]

        # Get action arguments for action id.
        # Look over this if stuff doesnt work later
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # Weird?

        return actions.FunctionCall(act_id, act_args)

    """ Record data for training """
    def record_step(self, obs, action, reward, new_obs):
        self.replay.add([obs, action, reward, new_obs])

    """ Train agent """
    def update(self):

        # Get the records from episode
        experiences = np.array(self.replay.get_all())
        num_experiences = len(experiences)

        # Compute R, value of last observation
        last_obs = experiences[-1][-1]
        minimap = np.array(last_obs['feature_minimap'], dtype = np.float32)
        minimap = np.expand_dims(minimap, axis = 0)
        screen = np.array(last_obs['feature_screen'], dtype = np.float32)
        screen = np.expand_dims(screen, axis = 0)
        info = np.zeros([1, NUM_ACTIONS], dtype = np.float32)
        info[0, last_obs['available_actions']] = 1

        feed_dict = {self.minimap: minimap,
                self.screen: screen,
                self.non_spatial: info}
        R = self.tf_session.run(self.value, feed_dict)[0]

        # Compute targets and masks
        minimaps = []
        screens = []
        non_spatials = []

        value_target = np.zeros([num_experiences], dtype=np.float32)
        value_target[-1] = R

        valid_spatial_action = np.zeros([num_experiences], dtype=np.float32)
        spatial_action_selected = np.zeros([num_experiences, SCREEN_SIZE[0]**2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([num_experiences, NUM_ACTIONS], dtype=np.float32)
        non_spatial_action_selected = np.zeros([num_experiences, NUM_ACTIONS], dtype=np.float32)

        experiences = experiences[::-1]
        for i, [obs, action, reward, next_obs] in enumerate(experiences):
            minimap = np.array(obs['feature_minimap'], dtype=np.float32)
            minimap = np.expand_dims(minimap, axis=0)
            screen = np.array(obs['feature_screen'], dtype=np.float32)
            screen = np.expand_dims(screen, axis=0)
            info = np.zeros([1, NUM_ACTIONS], dtype=np.float32)
            info[0, obs['available_actions']] = 1

            minimaps.append(minimap)
            screens.append(screen)
            non_spatials.append(info)

            act_id = action.function
            act_args = action.arguments

            value_target[i] = reward + 0.99 * value_target[i-1]

            valid_actions = obs["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * SCREEN_SIZE[0] + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        non_spatials = np.concatenate(non_spatials, axis=0)

        # Train
        feed = {self.minimap: minimaps,
            self.screen: screens,
            self.non_spatial: non_spatials,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected}
        self.tf_session.run(self.train_op, feed_dict=feed)

    """ Setup agent """
    def setup(self, tf_session):
        self.tf_session = tf_session
        tf_session.run(tf.global_variables_initializer())

    """ Reset agent """
    def reset(self):
        self.episodes += 1
        self.replay.clear()
