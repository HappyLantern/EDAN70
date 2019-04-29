import tensorflow as tf
from tensorflow.contrib import layers
from pysc2.lib import actions
from pysc2.lib import features
from expreplay import ExpReplay
import numpy
from utils import preprocess_minimap, preprocess_screen

NUM_SCREEN_FEATURES = len(features.SCREEN_FEATURES)
NUM_MINIMAP_FEATURES = len(features.MINIMAP_FEATURES)
SCREEN_SIZE = (84, 84)
MINIMAP_SIZE = (64, 64)
NUM_ACTIONS = len(actions.FUNCTIONS)


class TestAgent():

    def __init__(self):
        print("Create network here")
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        learning_rate = 1e-4
        self.gamma = 0.99
        self.replay = ExpReplay(10000)

        self.minimap = tf.placeholder(shape=[None, NUM_MINIMAP_FEATURES, *MINIMAP_SIZE],
                                      dtype=tf.float32)
        self.screen = tf.placeholder(shape=[None, NUM_SCREEN_FEATURES, *SCREEN_SIZE],
                                     dtype=tf.float32)
        self.info = tf.placeholder(shape=[None, NUM_ACTIONS],
                                   dtype=tf.float32)
        self.screen_processed = preprocess_screen(
            self.screen)

        self.minimap_processed = preprocess_minimap(
            self.minimap)

        self.mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]),
                                    num_outputs=16,
                                    kernel_size=5,
                                    stride=1,
                                    scope='mconv1')
        self.mconv2 = layers.conv2d(self.mconv1,
                                    num_outputs=32,
                                    kernel_size=3,
                                    stride=1,
                                    scope='mconv2')
        self.sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]),
                                    num_outputs=16,
                                    kernel_size=5,
                                    stride=1,
                                    scope='sconv1')
        self.sconv2 = layers.conv2d(self.sconv1,
                                    num_outputs=32,
                                    kernel_size=3,
                                    stride=1,
                                    scope='sconv2')
        self.info_fc = layers.fully_connected(layers.flatten(self.info),
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
        self.spatial_action = tf.nn.softmax(layers.flatten(self.spatial_action))

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
        function_id = numpy.random.choice(
            timestep.observation.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)

    def record_step(self, timesteps0, actions, timesteps1):
        print("Save here")
        self.replay.add([timesteps0, actions, timesteps1])

    def update(self):
        print("Update agent here")

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1
