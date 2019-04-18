import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features
from expreplay import ExpReplay
import numpy

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

        self.minimap = tf.placeholder(shape = [None, 64, 64], dtype = tf.float32)
        self.screen = tf.placeholder(shape = [None, 64, 64], dtype = tf.float32)
        self.info = tf.placeholder(shape = [None, len(actions.FUNCTIONS)], dtype = tf.float32)

        mconv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]),
            num_outputs=16,
            kernel_size=5,
            stride=1,
            scope='mconv1')
        mconv2 = layers.conv2d(mconv1,
            num_outputs=32,
            kernel_size=3,
            stride=1,
            scope='mconv2')
        sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]),
            num_outputs=16,
            kernel_size=5,
            stride=1,
            scope='sconv1')
        sconv2 = layers.conv2d(sconv1,
            num_outputs=32,
            kernel_size=3,
            stride=1,
            scope='sconv2')
        info_fc = layers.fully_connected(layers.flatten(self.info),
            num_outputs=256,
            activation_fn=tf.tanh,
            scope='info_fc')

        # Compute spatial actions
        feat_conv = tf.concat([mconv2, sconv2], axis=3)
            spatial_action = layers.conv2d(feat_conv,
                num_outputs=1,
                kernel_size=1,
                stride=1,
                activation_fn=None,
                scope='spatial_action')
        spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

        # Compute non spatial actions and value
        feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
        feat_fc = layers.fully_connected(feat_fc,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope='feat_fc')
        non_spatial_action = layers.fully_connected(feat_fc,
            num_outputs=num_action,
            activation_fn=tf.nn.softmax,
            scope='non_spatial_action')
        value = tf.reshape(layers.fully_connected(feat_fc,
            num_outputs=1,
            activation_fn=None,
            scope='value'), [-1])
#        self.output =
        def step(self, obs):
            """Do action here"""
            self.steps += 1
            self.reward += obs.reward
            function_id = numpy.random.choice(obs.observation.available_actions)
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
