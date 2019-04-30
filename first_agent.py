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

    def __init__(self):
        print("\n" + "-+"*40, "\nCreate network here\n" + "-+"*40, "\n")
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        learning_rate = 1e-4
        self.gamma = 0.99
        self.replay = ExpReplay(10000)

        self.minimap = tf.placeholder(shape=[None, NUM_MINIMAP_FEATURES, *MINIMAP_SIZE],
                                      dtype=tf.int32)
        self.screen = tf.placeholder(shape=[None, NUM_SCREEN_FEATURES, *SCREEN_SIZE],
                                     dtype=tf.int32)
        self.non_spatial = tf.placeholder(shape=[None, NUM_ACTIONS],
                                          dtype=tf.float32)
        self.screen_processed = preprocess_screen(
            self.screen)

        self.minimap_processed = preprocess_minimap(
            self.minimap)

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
        self.info_fc = layers.fully_connected(layers.flatten(self.non_spatial),
                                              num_outputs=256,
                                              activation_fn=tf.tanh,
                                              scope='info_fc')

        # Compute spatial actions
        self.feat_conv = tf.concat([self.mconv2, self.sconv2], axis=3)
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

    def step(self, timestep):
        """Do action here"""
        self.steps += 1
        self.reward += timestep.reward

        observation = timestep.observation
        screen_features = observation.feature_screen
        screen_features = np.expand_dims(screen_features, axis=0)
        minimap_features = observation.feature_minimap
        minimap_features = np.expand_dims(minimap_features, axis=0)
        available_actions = observation.available_actions
        action_mask = np.zeros(NUM_ACTIONS, dtype=np.int32)
        action_mask[available_actions] = 1
        action_mask = np.expand_dims(action_mask, axis=0)

        feed_dict = {self.minimap: minimap_features,
                     self.screen: screen_features,
                     self.non_spatial: action_mask}
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed_dict)
        #print("Non_spatial_action:", non_spatial_action, "\nSpatial_action:", spatial_action)

        # TODO: Below is copy pasted; might be wrong
        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel()
        valid_actions = timestep.observation.available_actions
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        target = [int(target // 64), int(target % 64)] #TODO: Changed from 84 to 64
        

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful

        # function_id = np.random.choice(
        #    timestep.observation.available_actions)
        # args = [[np.random.randint(0, size) for size in arg.sizes]
        #        for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(act_id, act_args)

    def record_step(self, timesteps0, actions, timesteps1):
        #print("Save here")
        self.replay.add([timesteps0, actions, timesteps1])

    def update(self):
        #print("Update agent here")
        pass

    def setup(self, obs_spec, action_spec, sess):
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.sess = sess
        sess.run(tf.global_variables_initializer())


    def reset(self):
        self.episodes += 1