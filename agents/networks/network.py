"""Neural networks that output value estimates for actions, given a state."""

import numpy as np
import tensorflow as tf
from pysc2.lib import features
from agents.preprocessing import preprocess_spatial_features

SCREEN_FEATURES = features.SCREEN_FEATURES
NUM_ACTIONS = 3
NUM_FEATURES = len(SCREEN_FEATURES)

class Network(object):
    """Uses feature_screen.player_relative to assign q value to movements."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="DQN"):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # trackers
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")
                
            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # network architecture
            self.inputs = tf.placeholder(
                tf.int32,
                [None, NUM_FEATURES, *self.spatial_dimensions],
                name="inputs")

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            self.screen_processed = preprocess_spatial_features(self.inputs, screen=True)

            self.conv1 = tf.layers.conv2d(
                inputs=self.screen_processed,
                filters=16,
                kernel_size=[5, 5],
                strides=[1, 1],
                padding="SAME",
                activation=tf.nn.relu,
                name="conv1")

            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1,
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                activation=tf.nn.relu,
                name="conv2")

            self.output = tf.layers.conv2d(
                inputs=self.conv2,
                filters=NUM_ACTIONS,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="output")

            self.flatten = tf.layers.flatten(self.output, name="flat")
            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.flatten, name="max")
            self.mean_q = tf.reduce_mean(self.flatten, name="mean")

            # optimization: MSE between state predicted Q and target Q
            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod((84, 84, NUM_ACTIONS))],
                name="actions")

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")

            self.prediction = tf.reduce_sum(
                tf.multiply(self.flatten, self.actions),
                axis=1,
                name="prediction")

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)