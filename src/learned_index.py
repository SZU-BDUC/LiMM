# -*- coding:utf-8 -*-
import numpy as np
import time
import os
import math
import tensorflow.compat.v1 as tf
from six.moves import xrange
import pickle

from data_process.hexagon import label_point
from data_process.dis_latlon import get_bear, distance_meter, distance_point_to_segment, destination
from math import radians

class DataSet(object):

    def __init__(self, keys, positions=None, num_positions=None):
        """Construct a DataSet.
        """

        assert keys.shape[0] == positions.shape[0], (
                'keys.shape: %s positions.shape: %s' % (keys.shape, positions.shape))

        self._num_keys = keys.shape[0]

        self._keys = np.array(keys)
        if positions is not None:
            self._positions = np.array(positions)
        else:
            self._keys = np.sort(keys)
            self._positions = np.arange(self._num_keys)

        if num_positions is not None:
            self._num_positions = num_positions
        else:
            if len(self._positions) == 0:
                self._num_positions = 0
            else:
                self._num_positions = self._positions[-1] + 1

        self._epochs_completed = 0
        self._index_in_epoch = 0

        if len(keys.shape) > 1:
            self._key_size = keys.shape[1]
        else:
            self._key_size = 1

        if len(keys) > 0:
            self._keys_mean = np.mean(keys)
            self._keys_std = np.std(keys)
        else:
            self._keys_mean = None
            self._keys_std = None

    @property
    def keys(self):
        return self._keys

    @property
    def positions(self):
        return self._positions

    @property
    def num_keys(self):
        return self._num_keys

    @property
    def num_positions(self):
        return self._num_positions

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def key_size(self):
        return self._key_size

    @property
    def keys_mean(self):
        return self._keys_mean

    @property
    def keys_std(self):
        return self._keys_std

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_keys:
            # Finished epoch
            self._epochs_completed += 1
            if shuffle:
                # Shuffle the data
                perm = np.arange(self._num_keys)
                np.random.shuffle(perm)
                self._keys = self._keys[perm]
                self._positions = self._positions[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_keys
        end = self._index_in_epoch
        return self._keys[start:end], self._positions[start:end]

    def reset_epoch(self):
        self._index_in_epoch = 0


def create_train_validate_data_sets(data_set, validation_size=0):
  """Creates training and validation data sets.
  """

  #Shuffle the keys and positions by same permutation
  perm = np.arange(data_set.num_keys)
  np.random.shuffle(perm)
  keys = data_set.keys[perm]
  positions = data_set.positions[perm]

  if not 0 <= validation_size <= len(keys):
    raise ValueError(
        "Validation size should be between 0 and {}. Received: {}."
        .format(len(keys), validation_size))

  validation_keys = keys[:validation_size]
  validation_positions = positions[:validation_size]
  train_keys = keys[validation_size:]
  train_positions = positions[validation_size:]

  train = DataSet(np.reshape(train_keys,[-1,1]), train_positions)
  validation = DataSet(validation_keys, validation_positions)


  class DataSets(object):
    pass

  data_sets = DataSets()
  data_sets.train = train
  data_sets.validate = validation
  return data_sets


##
def generate_uniform_floats(num_keys=100000, key_range=[0.0, 1.0], iseed=None):
    """Generate a DataSet of uniform floating points.
    """

    np.random.seed(iseed)
    keys = np.random.random(num_keys)
    keys = (key_range[1] - key_range[0]) * keys + key_range[0]

    keys = np.sort(keys)
    positions = np.arange(num_keys)

    return DataSet(keys=keys, positions=positions)


def generate_normal_floats(num_keys=100000, mean=0, std=1.0, iseed=None):
    """Generate a DataSet of normallaly distributed floating points.
    """

    np.random.seed(iseed)
    keys = np.random.normal(loc=mean, scale=std, size=num_keys)

    keys = np.sort(keys)
    positions = np.arange(num_keys)

    return DataSet(keys=keys, positions=positions)


def load_keys_npy(dir="./test_data", fname="uniform_floats.npy"):
    """Load keys from .npy file"""

    keys = np.load(os.path.join(dir, fname))
    keys = np.unique(keys)  # Unique returns sorted data
    positions = np.arange(len(keys))

    return DataSet(keys=keys, positions=positions)


##


class RmiSimple(object):
    """ Implements the simple "Recursive-index model" described in the paper
        'The Case for Learned Index Structures', which can be found at
        [Kraska et al., 2017](http://arxiv.org/abs/1712.01208)
        ([pdf](http://arxiv.org/pdf/1712.01208.pdf)).

        The first stage is a fully connected neural network with any number
        (>=0) of hidden layers. Each second stage model is a single-variable
        linear regression.

        At model creation, the user can choose the widths of the
        hidden layers and the number of models ("experts") used in
        stage 2.
    """

    def __init__(self,
                 data_set,
                 hidden_layer_widths=[16, 16],
                 num_experts=10,
                 learning_rates=[0.1, 0.1],
                 max_steps=[1000, 1000],
                 batch_sizes=[1000, 1000],
                 model_save_dir='tf_checkpoints'):
        """Initializes the Recursive-index model

        Args:
            data_set: object of type DataSet, which the model will train on
            hidden layer_widths: list of hidden layer widths (use empty list
                                 for zero hidden layers)
            num_experts: number of models ("experts") used in stage 2
            learning_rates: list (length=2) of learning rates for each stage
            max_steps: list (length=2) of maximum number of training steps for each stage
            batch_sizes: list (length=2) of batch training sizes for each stage
            model_save_dir: Name of directory to save model

        """

        # Initialize from input parameters
        self._data_set = data_set
        self.hidden_layer_widths = hidden_layer_widths
        self.num_experts = num_experts
        self.learning_rates = learning_rates
        self.max_steps = max_steps
        self.batch_sizes = batch_sizes
        self.model_save_dir = model_save_dir

        # Decide which optimized inference function to use, based on
        # number of hidden layers.

        num_hidden_layers = len(self.hidden_layer_widths)

        if num_hidden_layers == 0:
            self.run_inference = self._run_inference_numpy_0_hidden
        elif num_hidden_layers == 1:
            self.run_inference = self._run_inference_numpy_1_hidden
        elif num_hidden_layers == 2:
            self.run_inference = self._run_inference_numpy_2_hidden
        else:
            self.run_inference = self._run_inference_numpy_n_hidden

        # Store prediction errors for each expert
        # Fill these values using self.calc_min_max_errors()
        self.max_error_left = None
        self.max_error_right = None
        self.min_predict = None
        self.max_predict = None
        self.min_pos = None
        self.max_pos = None

        self._initialize_errors()

        # Define variables to stored trained tensor variables
        # (e.g. weights and biases).
        # These are used to run inference faster with numpy
        # rather than with TensorFlow.

        self.hidden_w = [None] * num_hidden_layers
        self.hidden_b = [None] * num_hidden_layers
        self.linear_w = None
        self.linear_b = None
        self.stage_2_w = None
        self.stage_2_b = None
        self._expert_factor = None

        # Pre-calculate some normalization and computation constants,
        # so that they are not repeatedly calculated later.

        # Normalize using mean and dividing by the standard deviation
        self._keys_mean = self._data_set.keys_mean
        self._keys_std_inverse = 1.0 / self._data_set.keys_std
        # Normalize further by dividing by 2*sqrt(3), so that
        # a uniform distribution in the range [a,b] would transform
        # to a uniform distribution in the range [-0.5,0.5]
        self._keys_norm_factor = 0.5 / np.sqrt(3)
        # Precalculation for expert = floor(stage_1_pos * expert_factor)
        self._expert_factor = self.num_experts / self._data_set.num_positions

    def new_data(self, data_set):
        """Changes the data set used for training. For example, this function should
           be called after a large number of inserts are performed.

        Args:
            data_set: type DataSet, replaces current data_set with new data_set
        """

        self._data_set = data_set

        # Normalize using mean and dividing by the standard deviation
        self._keys_mean = self._data_set.keys_mean
        self._keys_std_inverse = 1.0 / self._data_set.keys_std
        # Normalize further by dividing by 2*sqrt(3), so that
        # a uniform distribution in the range [a,b] would transform
        # to a uniform distribution in the range [-0.5,0.5]
        self._keys_norm_factor = 0.5 / np.sqrt(3)
        # Precalculation for expert = floor(stage_1_pos * expert_factor)
        self._expert_factor = self.num_experts / self._data_set.num_positions

    def _setup_placeholder_inputs(self, batch_size):
        """Create placeholder tensors for inputing keys and positions.

        Args:
            batch_size: Batch size.

        Returns:
            keys_placeholder: Keys placeholder tensor.
            labels_placeholder: Labels placeholder tensor.
        """

        # The first dimension is None for both placeholders in order
        # to handle variable batch sizes
        with tf.name_scope("placeholders"):
            keys_placeholder = tf.placeholder(tf.float32, shape=(None, self._data_set.key_size), name="keys")
            labels_placeholder = tf.placeholder(tf.int64, shape=(None), name="labels")
        return keys_placeholder, labels_placeholder

    def _fill_feed_dict(self, keys_pl, labels_pl, batch_size=100, shuffle=True):
        """ Creates a dictionary for use with TensorFlow's feed_dict

        Args:
            keys_pl: TensorFlow (TF) placeholder for keys,
                     created from self._setup_placeholder_inputs().
            labels_pl: TF placeholder for labels (i.e. the key positions)
                     created from self._setup_placeholder_inputs().
            batch_size: integer size of batch
            shuffle: whether or not to shuffle the data
                     Note: shuffle=Flase can be useful for debugging

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.

        keys_feed, labels_feed = self._data_set.next_batch(batch_size, shuffle)
        feed_dict = {
            keys_pl: keys_feed,
            labels_pl: labels_feed,
        }
        return feed_dict

    def _setup_inference_stage_1(self, keys):
        """Set up Stage 1 inference.

        Args:
            keys: Tensorflow placeholder for keys

        Returns:
            pos_stage_1: Output tensor that predicts key position

        """

        # All Stage 1 operations should be in 'stage_1' name_Scope
        with tf.name_scope('stage_1'):

            keys_std = self._data_set.keys_std
            keys_mean = self._data_set.keys_mean
            key_size = self._data_set.key_size

            hidden_widths = self.hidden_layer_widths

            # Normalize
            with tf.name_scope('normalize'):

                keys = tf.cast(keys, dtype=tf.float64)

                # Normalize using mean and standard deviation
                keys_normed = tf.scalar_mul(tf.constant(1.0 / keys_std),
                                            tf.subtract(keys, tf.constant(keys_mean)))

                # Normalize further by dividing by 2*sqrt(3), so that
                # a uniform distribution in the range [a,b] would transform
                # to a uniform distribution in the range [-0.5,0.5]

                keys_normed = tf.scalar_mul(tf.constant(0.5 / np.sqrt(3)),
                                            keys_normed)

            # All hidden layers
            tf_output = keys_normed  # previous output
            output_size = key_size  # previous output size
            for layer_idx in range(0, len(hidden_widths)):
                tf_input = tf_output  # get current inputs from previous outputs
                input_size = output_size
                output_size = hidden_widths[layer_idx]
                name_scope = "hidden_" + str(layer_idx + 1)  # Layer num starts at 1
                with tf.name_scope(name_scope):
                    weights = tf.Variable(
                        tf.truncated_normal([input_size, output_size],
                                            stddev=1.0 / math.sqrt(float(input_size)),
                                            dtype=tf.float64),
                        name='weights',
                        dtype=tf.float64)
                    biases = tf.Variable(tf.zeros([output_size], dtype=tf.float64),
                                         name='biases',
                                         dtype=tf.float64)
                    tf_output = tf.nn.relu(tf.matmul(tf_input, weights) + biases)

            # Linear
            with tf.name_scope('linear'):
                weights = tf.Variable(
                    tf.truncated_normal([output_size, 1],
                                        stddev=1.0 / math.sqrt(float(output_size)),
                                        dtype=tf.float64),
                    name='weights')
                biases = tf.Variable(tf.zeros([1], dtype=tf.float64),
                                     name='biases')

                pos_stage_1 = tf.matmul(tf_output, weights) + biases

                if (key_size == 1):
                    pos_stage_1 = tf.reshape(pos_stage_1, [-1])

                # At this point we want the model to have produced
                # output in the range [-0.5, 0.5], but we want the
                # final output to be in the range [0,N), so we need
                # to add 0.5 and multiply by N.
                # Doing normalization this way can effect how
                # the learning rates scale with N, so we should
                # consider doing this normalization outside of
                # the Tensflow pipeline.
                pos_stage_1 = tf.scalar_mul(tf.constant(self._data_set.num_positions,
                                                        dtype=tf.float64),
                                            tf.add(pos_stage_1,
                                                   tf.constant(0.5, dtype=tf.float64)))

                pos_stage_1 = tf.identity(pos_stage_1, name="pos")

        return pos_stage_1

    def _setup_loss_stage_1(self, pos_stage_1, pos_true):
        """Calculates the loss from the keys and positions, for Stage 1.
        Args:
        pos_stage_1: int64 tensor with shape [batch_size, 1].
                     The position predicted in stage 1
        pos_true: int64 tensor wiht shape [batch_size].
                  The true position for the key.
        Returns:
        loss: Loss tensor, using mean_squared_error.
        """
        labels = tf.to_int64(pos_true)
        loss = tf.losses.mean_squared_error(
            labels=pos_true,
            predictions=pos_stage_1)

        return loss

    def _setup_training_stage_1(self, loss):
        """Sets up the TensorFlow training operations for Stage 1.

        Args:
            loss: loss tensor, from self._setup_loss_stage_1()

        Returns:
            train_op: the TensorFlow operation for training Stage 1.
        """

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        # Create optimizer with the given learning rate.
        # AdamOptimizer is used, but other other optimizers could
        # have been chosen (e.g. the commented-out examples).
        optimizer = tf.train.AdamOptimizer(self.learning_rates[0])
        # optimizer = tf.train.AdadeltaOptimizer(self.learning_rates[0])
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rates[0])

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def _setup_inference_stage_2(self, keys, pos_stage_1):
        """Set up Stage 2 inference.

        Args:
            keys: TensorFlow placeholder for keys
            pos_stage_1: tensor, output of Stage 1 inference
        Returns:
            pos_stage_2: tensor, output of Stage 2 inference
        """

        max_index = self._data_set.num_positions

        # Stage 2
        with tf.name_scope('stage_2'):
            keys_std = self._data_set.keys_std
            keys_mean = self._data_set.keys_mean

            keys = tf.squeeze(keys, 1)
            keys = tf.identity(keys, name='key')
            keys = tf.cast(keys, dtype=tf.float64)

            # Normalize using mean and standard deviation
            keys_normed = tf.scalar_mul(tf.constant(1.0 / keys_std),
                                        tf.subtract(keys, tf.constant(keys_mean)))

            # Normalize further by dividing by 2*sqrt(3), so that
            # a uniform distribution in the range [a,b] would transform
            # to a uniform distribution in the range [-0.5,0.5]

            keys_normed = tf.scalar_mul(tf.constant(0.5 / np.sqrt(3)),
                                        keys_normed)

            # Calculate which expert to use
            expert_index = tf.to_int32(
                tf.floor(
                    tf.scalar_mul(tf.constant(self._expert_factor, dtype=tf.float64),
                                  pos_stage_1)))

            # Ensure that expert_index is within range [0,self.num_experts)
            expert_index = tf.maximum(tf.constant(0), expert_index)
            expert_index = tf.minimum(tf.constant(self.num_experts - 1), expert_index)
            expert_index = tf.identity(expert_index, name="expert_index")

            # Explicitly handle batches
            num_batches = tf.shape(pos_stage_1)[0]
            num_batches = tf.identity(num_batches, name="num_batches")
            expert_index_flat = (tf.reshape(expert_index, [-1])
                                 + tf.range(num_batches) * self.num_experts)
            expert_index_flat = tf.identity(expert_index_flat, name="expert_index_flat")

            # This version uses tf.unsroted_segment_sum
            gates_flat = tf.unsorted_segment_sum(
                tf.ones_like(expert_index_flat),
                expert_index_flat,
                num_batches * self.num_experts)
            gates = tf.reshape(gates_flat, [num_batches, self.num_experts],
                               name="gates")

            # This version uses SparseTensor, and could potential replace the
            # previous block of code, but it doesn't work yet
            #
            # expert_index_flat = tf.reshape(expert_index_flat,[-1,1])
            # gates_flat = tf.SparseTensor(tf.cast(expert_index_flat,dtype=tf.int64),
            #                             tf.ones([self.num_experts]),
            #                             dense_shape=[self.num_experts*num_batches,1])
            # gates = tf.sparse_reshape(gates_flat, [num_batches, tf.constant(self.num_experts)],
            #                          name="gates")
            # gates = tf.sparse_tensor_to_dense(gates)

            # Name the gates for later access
            gates = tf.cast(gates, dtype=tf.float64)
            gates = tf.identity(gates, name="gates")

            # Normalize variable weights and biases
            weights = tf.Variable(
                tf.truncated_normal([self.num_experts],
                                    mean=1.0 * max_index,
                                    stddev=0.5 * max_index,
                                    dtype=tf.float64),
                name='weights')

            biases = tf.Variable(tf.zeros([self.num_experts], dtype=tf.float64),
                                 name='biases')

            # Dot-product gates with weights and biases,
            # to only use one expert at a time.
            gated_weights = tf.multiply(gates, weights)
            gated_biases = tf.multiply(gates, biases)
            gated_weights_summed = tf.reduce_sum(gated_weights, axis=1)
            gated_biases_summed = tf.reduce_sum(gated_biases, axis=1)

            # Name the variables for later access
            gated_weights = tf.identity(gated_weights, name="gated_weights")
            gated_biases = tf.identity(gated_biases, name="gated_biases")
            gated_weights_summed = tf.identity(gated_weights_summed, name="gated_weights_summed")
            gated_biases_summed = tf.identity(gated_biases_summed, name="gated_biases_summed")

            # Do the linear regression to predict the key position
            pos_stage_2 = tf.add(tf.multiply(keys_normed, gated_weights_summed), gated_biases_summed)
            pos_stage_2 = tf.identity(pos_stage_2, name="pos")

        # Returns the predicted position for Stage 2
        return pos_stage_2

    def _setup_loss_stage_2(self, pos_stage_2, pos_true):
        """Calculates the loss from the keys and positions, for Stage 2.

        Args:
            pos_stage_2: int64 tensor with shape [batch_size, 1].
                         The position predicted in stage 1
            pos_true: int64 tensor wiht shape [batch_size].
                      The true position for the key.
        Returns:
            loss: Loss tensor, using mean_squared_error.
        """
        # Stage 2
        with tf.name_scope('stage_2'):
            labels = tf.to_int64(pos_true)
            loss = tf.losses.mean_squared_error(
                labels=pos_true,
                predictions=pos_stage_2)

        return loss

    def _setup_training_stage_2(self, loss):
        """Sets up the TensorFlow training operations for Stage 2.

        Args:
            loss: loss tensor, from self._setup_loss_stage_2()

        Returns:
            train_op: the TensorFlow operation for training Stage 2.
        """

        # Stage 2
        with tf.name_scope('stage_2'):
            # Add a scalar summary for the snapshot loss.
            tf.summary.scalar('loss', loss)

            # Create optimizer with the given learning rate.
            # Uses AdamOptimizer, but others could be considered
            # (e.g. see commented-out examples)
            optimizer = tf.train.AdamOptimizer(self.learning_rates[1])
            # optimizer = tf.train.AdadeltaOptimizer(self.learning_rates[1])
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rates[1])

            # Create a variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Get list of variables needed to train stage 2
            variables_stage_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stage_2')
            # Use the optimizer to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training step.
            train_op = optimizer.minimize(loss, global_step=global_step, var_list=variables_stage_2)

            return train_op

    def run_training(self,
                     batch_sizes=None,
                     max_steps=None,
                     learning_rates=None,
                     model_save_dir=None):
        """Train both Stage 1 and Stage 2 (in order)

        Args:
            batch_sizes: list (length=2) of batch sizes for the two Stages.
                default=None (use the model's  self.batch_sizes)
            max_steps: list (length=2) of number of training steps for the two Stages.
                default=None (use the model's self.max_steps)
            learning_rates: list (length=2) of learning rates for the two Stages.
                default=None (use the model's self.learning_rates)
            model_save_dir: Name of directory to save the model
                default=None (use the model's self.model_save_dir)

        Returns:
            No output, but prints training information to stdout.
        """

        # First update model with new batch_sizes, learning_rates, max_steps,
        # and model_save_dir
        if batch_sizes is not None:
            self.batch_sizes = batch_sizes
        if learning_rates is not None:
            self.learning_rates = learning_rates
        if max_steps is not None:
            self.max_steps = max_steps
        if model_save_dir is not None:
            self.model_save_dir = model_save_dir

        # Reset the default graph
        tf.reset_default_graph()

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            ## Stage 1

            # Generate placeholders for the images and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(self.batch_sizes[0])

            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            # Add to the Graph the Ops for loss calculation.
            loss_s1 = self._setup_loss_stage_1(pos_stage_1, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op_s1 = self._setup_training_stage_1(loss_s1)

            # Currently no need for Summaries, but could add this later
            # Build the summary Tensor based on the TF collection of Summaries.
            # summary = tf.summary.merge_all()

            ## Stage 2

            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1)

            # Add to the Graph the Ops for loss calculation.
            loss_s2 = self._setup_loss_stage_2(pos_stage_2, labels_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op_s2 = self._setup_training_stage_2(loss_s2)

            ## Done with Stage definitions

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Could use a SummaryWrite in future implementation
            # Instantiate a SummaryWriter to output summaries and the Graph.
            # summary_writer = tf.summary.FileWriter(model_save_dir, sess.graph)

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)

            ## Train Stage 1
            print("Stage 1 Training:")

            training_start_time = time.time()

            # Start the training loop.
            for step in xrange(self.max_steps[0]):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of keys and labels
                # for this particular training step.
                feed_dict = self._fill_feed_dict(keys_placeholder,
                                                 labels_placeholder,
                                                 batch_size=self.batch_sizes[0])

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op_s1, loss_s1],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec, total %.3f secs)' % (
                    step, np.sqrt(loss_value), duration, time.time() - training_start_time))
                    # Could write summary info in future implementation.
                    # Update the events file.
                    # summary_str = sess.run(summary, feed_dict=feed_dict)
                    # summary_writer.add_summary(summary_str, step)
                    # summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 10000 == 0 and (step + 1) != self.max_steps[0]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_1.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                if (step + 1) == self.max_steps[0]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_1.ckpt')
                    saver.save(sess, checkpoint_file)

            ## Train Stage 2
            print("\nStage 2 Training:")

            # Start the training loop.
            for step in xrange(self.max_steps[1]):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of keys and labels
                # for this particular training step.
                feed_dict = self._fill_feed_dict(keys_placeholder,
                                                 labels_placeholder,
                                                 batch_size=self.batch_sizes[1])

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op_s2, loss_s2],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec, total %.3f secs)' % (
                    step, np.sqrt(loss_value), duration, time.time() - training_start_time))
                    # Could write summary info in future implementation.
                    # Update the events file.
                    # summary_str = sess.run(summary, feed_dict=feed_dict)
                    # summary_writer.add_summary(summary_str, step)
                    # summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 10000 == 0 and (step + 1) != self.max_steps[1]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_2.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                if (step + 1) == self.max_steps[1]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_2.ckpt')
                    saver.save(sess, checkpoint_file)

    def _run_inference_tensorflow(self, keys):
        """Run inference using TensorFlow checkpoint

        Args:
            keys: numpy array of one or more keys

        Returns:
            pos_stage_2: numpy array of predicted position for each key.
            expert: numpy array of expert used for each key.
        """

        batch_size = keys.shape[0]

        # Reset the default graph
        tf.reset_default_graph()

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            ## Stage 1

            # Generate placeholders for the keys and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(batch_size)

            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2

            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1)

            ## Done with Stage definitions

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Run the Op to initialize the variables.
            sess.run(init)

            # Load trained variables
            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir, "stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, checkpoint_file)

            # Fill a feed dictionary with keys
            feed_dict = {keys_placeholder: keys}

            # Get the expert for each key
            expert_index = sess.graph.get_tensor_by_name("stage_2/expert_index:0")
            experts = sess.run(expert_index, feed_dict=feed_dict)

            # Get the predicted position for each key
            stage_2_out = sess.graph.get_tensor_by_name("stage_2/pos:0")
            pos = sess.run(stage_2_out, feed_dict=feed_dict)

        return (pos, experts)

    def inspect_inference_steps(self, keys):
        """Run inference using TensorFlow, and print out important tensor
           values. Can be useful for debugging.

        Args:
            keys: numpy array of one or more keys

        Returns:
           Prints the values of several model tensors to stdout.
        """

        batch_size = keys.shape[0]

        # Reset the default graph
        tf.reset_default_graph()

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            ## Stage 1

            # Generate placeholders for the keys and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(batch_size)

            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2

            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1)

            ## Done with Stage definitions

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Run the Op to initialize the variables.
            sess.run(init)

            # Load trained variables
            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir, "stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, checkpoint_file)

            # Fill a feed dictionary with keys
            feed_dict = {keys_placeholder: keys}

            # Print the values of tensors used in the model

            print("Stage 1 position predictions (one per batch):")
            print(sess.run(pos_stage_1, feed_dict=feed_dict))

            print("Expert Index (one per batch):")
            expert_index = sess.graph.get_tensor_by_name("stage_2/expert_index:0")
            print(sess.run(expert_index, feed_dict=feed_dict))

            print("Expert Index Flat (all batches):")
            expert_index_flat = sess.graph.get_tensor_by_name("stage_2/expert_index_flat:0")
            print(sess.run(expert_index_flat, feed_dict=feed_dict))

            print("Gate vector (one per batch):")
            gates = sess.graph.get_tensor_by_name("stage_2/gates:0")
            print(sess.run(gates, feed_dict=feed_dict))

            print("Gate vector times weights (one per batch):")
            gated_weights = sess.graph.get_tensor_by_name("stage_2/gated_weights:0")
            print(sess.run(gated_weights, feed_dict=feed_dict))

            print("Gate vector times weights summed (one per batch):")
            gated_weights_summed = sess.graph.get_tensor_by_name("stage_2/gated_weights_summed:0")
            print(sess.run(gated_weights_summed, feed_dict=feed_dict))

            print("Gate vector times biases (one per batch):")
            gated_biases = sess.graph.get_tensor_by_name("stage_2/gated_biases:0")
            print(sess.run(gated_biases, feed_dict=feed_dict))

            print("Gate vector times biases summed (one per batch):")
            gated_biases_summed = sess.graph.get_tensor_by_name("stage_2/gated_biases_summed:0")
            print(sess.run(gated_biases_summed, feed_dict=feed_dict))

            print("Key (one per batch):")
            key = sess.graph.get_tensor_by_name("stage_2/key:0")
            print(sess.run(key, feed_dict=feed_dict))

            print("Stage 2 position prediction = w*key + b (one per batch):")
            stage_2_out = sess.graph.get_tensor_by_name("stage_2/pos:0")
            print(sess.run(stage_2_out, feed_dict=feed_dict))

    def get_weights_from_trained_model(self):
        """Retrieves weights and biases from TensorFlow checkpoints.
           Stores the weights and biases in class member variables, to be used
           for faster inference calculations (such as using numpy).

        Args:
            -

        Returns:
            -
        """

        # Reset the default graph
        tf.reset_default_graph()

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the keys and labels.
            batch_size = 1
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(1)
            ## Stage 1

            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2

            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1)

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Run the Op to initialize the variables.
            sess.run(init)

            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir, "stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, checkpoint_file)

            # Get the weights and biases variables, and store them

            for layer_idx in range(0, len(self.hidden_layer_widths)):
                name_scope = "stage_1/hidden_" + str(layer_idx + 1)

                weights = sess.graph.get_tensor_by_name(name_scope + "/weights:0")
                self.hidden_w[layer_idx] = sess.run(weights)

                biases = sess.graph.get_tensor_by_name(name_scope + "/biases:0")
                self.hidden_b[layer_idx] = sess.run(biases)

            linear_w = sess.graph.get_tensor_by_name("stage_1/linear/weights:0")
            self.linear_w = sess.run(linear_w)

            linear_b = sess.graph.get_tensor_by_name("stage_1/linear/biases:0")
            self.linear_b = sess.run(linear_b)

            stage_2_w = sess.graph.get_tensor_by_name("stage_2/weights:0")
            self.stage_2_w = sess.run(stage_2_w)

            stage_2_b = sess.graph.get_tensor_by_name("stage_2/biases:0")
            self.stage_2_b = sess.run(stage_2_b)

    def time_inference_tensorflow(self, N=100):
        """Calculates time per inference using TensorFlow, not counting the time
           it takes to start a sessions and load the graph.

        Args:
            N: Number of time to run inference to get an average.

        Returns:
            Time (in seconds) to run inference on one batch.
        """

        # Only test batch_size of 1.
        # Future implementations should consider timing larger batches.

        batch_size = 1

        # Reset the default graph
        tf.reset_default_graph()

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            ## Stage 1

            # Generate placeholders for the images and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(batch_size)
            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2

            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1)

            ## Done with Stage definitions

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Run the Op to initialize the variables.
            sess.run(init)

            # Load checkpoint
            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir, "stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, checkpoint_file)

            # Time N inference steps

            start_time = time.time()
            for n in range(N):
                key = self._data_set.keys[n]

                # Fill a feed dictionary with the set of keys

                feed_dict = {keys_placeholder: [key]}

                expert_index = sess.graph.get_tensor_by_name("stage_2/expert_index:0")
                experts = sess.run(expert_index, feed_dict=feed_dict)

                stage_2_out = sess.graph.get_tensor_by_name("stage_2/pos:0")
                pos = sess.run(stage_2_out, feed_dict=feed_dict)

        return (time.time() - start_time) / N

    def calc_min_max_errors(self,
                            key_pos=None,
                            batch_size=10000):
        """Calculates the errors each Stage 2 expert makes in predicting the
           keys poistion. Inference is run on the full data set to get the errors.
           The calculated prediction errors are stored in class member variables.

        Args:
            key_pos: Numpy array of (key,position) pairs for which to calculate errors.
                     If key_pos==None, then all keys are used.
            batch_size: integer size of batches.

        Returns:
            -
        """

        if key_pos == None:  # Use all keys

            # Initialize errors
            self._initialize_errors()

            # Use all keys and positions
            keys = self._data_set.keys
            true_positions = self._data_set.positions
            num_keys = self._data_set.num_keys

        else:  # Use subset of keys

            # Only use keys and position specified by key_pos
            keys, true_positions = list(zip(*key_pos))
            keys = list(keys)
            true_positions = list(true_positions)
            num_keys = len(keys)

        # Calculate errors for each expert
        for step in range(0, num_keys, batch_size):

            positions, experts = self.run_inference(keys[step:(step + batch_size)])
            true_positions_batch = true_positions[step:(step + batch_size)]

            for idx in range(len(positions)):

                pos = np.round(positions[idx])
                expert = experts[idx]
                true_pos = true_positions_batch[idx]

                self.min_predict[expert] = np.minimum(self.min_predict[expert],
                                                      pos)
                self.max_predict[expert] = np.maximum(self.max_predict[expert],
                                                      pos)

                self.min_pos[expert] = np.minimum(self.min_pos[expert],
                                                  true_pos)
                self.max_pos[expert] = np.maximum(self.max_pos[expert],
                                                  true_pos)

                error = pos - true_pos
                if error > 0:
                    self.max_error_left[expert] = np.maximum(self.max_error_left[expert],
                                                             error)
                elif error < 0:
                    self.max_error_right[expert] = np.maximum(self.max_error_right[expert],
                                                              np.abs(error))

    def _initialize_errors(self):
        """Helper function that initializes all errors before call to

        Args:
            --

        Returns:
            --

        """

        # Initialize errors for each expert

        # The maximum left and right error for each expert
        self.max_error_left = np.zeros([self.num_experts])
        self.max_error_right = np.zeros([self.num_experts])

        # The minimum and maximum position predictions of each expert
        self.min_predict = (np.ones([self.num_experts]) * self._data_set.num_positions) - 1
        self.max_predict = np.zeros([self.num_experts])

        # The minimum and maximum true positions handled by each expert
        self.min_pos = (np.ones([self.num_experts]) * self._data_set.num_positions) - 1
        self.max_pos = np.zeros([self.num_experts])

    def _run_inference_numpy_0_hidden(self, keys):
        """Run inference using numpy, assuming 0 hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.

        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor

        out = np.matmul(keys, self.linear_w)
        out = np.add(out, self.linear_b)

        out = np.add(out, 0.5)
        out = np.multiply(out, self._data_set.num_positions)

        expert = np.multiply(out, self._expert_factor)
        expert = expert.astype(np.int32)  # astype() equivalent to floor() + casting
        expert = np.maximum(0, expert)
        expert = np.minimum(self.num_experts - 1, expert)

        out = np.multiply(keys, self.stage_2_w[expert])
        out = np.add(out, self.stage_2_b[expert])

        return (out, expert)

    def _run_inference_numpy_0_hidden_0_experts(self, keys):
        """Run inference using numpy, assuming 0 hidden layers in Stage 1.

        Args:
            keys: numpy array of keys.

        Returns:

            pos_stage_1: Position predictions for the keys.

        """

        # Do the same calculations found in self._setup_inference_stage_1()
        # and in self._setup_inference_stage_2(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor

        out = np.matmul(keys, self.linear_w)
        out = np.add(out, self.linear_b)

        out = np.add(out, 0.5)
        out = np.multiply(out, self._data_set.num_positions)

        return out

    def _run_inference_numpy_1_hidden(self, keys):
        """Run inference using numpy, assuming 1 hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.

        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor

        out = np.matmul(keys, self.hidden_w[0])
        out = np.add(out, self.hidden_b[0])
        out = np.maximum(0.0, out)

        out = np.matmul(out, self.linear_w)
        out = np.add(out, self.linear_b)

        out = np.add(out, 0.5)
        out = np.multiply(out, self._data_set.num_positions)

        expert = np.multiply(out, self._expert_factor)
        expert = expert.astype(np.int32)  # astype() equivalent to floor() + casting
        expert = np.maximum(0, expert)
        expert = np.minimum(self.num_experts - 1, expert)

        out = np.multiply(keys, self.stage_2_w[expert])
        out = np.add(out, self.stage_2_b[expert])

        return (out, expert)

    def _run_inference_numpy_2_hidden(self, keys):
        """Run inference using numpy, assuming 2 hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.

        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor

        out = np.matmul(keys, self.hidden_w[0])
        out = np.add(out, self.hidden_b[0])
        out = np.maximum(0.0, out)

        out = np.matmul(out, self.hidden_w[1])
        out = np.add(out, self.hidden_b[1])
        out = np.maximum(0.0, out)

        out = np.matmul(out, self.linear_w)
        out = np.add(out, self.linear_b)

        out = np.add(out, 0.5)
        out = np.multiply(out, self._data_set.num_positions)

        expert = np.multiply(out, self._expert_factor)
        expert = expert.astype(np.int32)  # astype() equivalent to floor() + casting
        expert = np.maximum(0, expert)
        expert = np.minimum(self.num_experts - 1, expert)

        out = np.multiply(keys, self.stage_2_w[expert])
        out = np.add(out, self.stage_2_b[expert])

        return (out, expert)

    def _run_inference_numpy_n_hidden(self, keys):
        """Run inference using numpy, assuming any number of hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.

        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor

        out = keys
        for layer_idx in range(0, len(self.hidden_layer_widths)):
            out = np.matmul(out, self.hidden_w[layer_idx])
            out = np.add(out, self.hidden_b[layer_idx])
            out = np.maximum(0.0, out)

        out = np.matmul(out, self.linear_w)
        out = np.add(out, self.linear_b)

        out = np.add(out, 0.5)
        out = np.multiply(out, self._data_set.num_positions)

        expert = np.multiply(out, self._expert_factor)
        expert = expert.astype(np.int32)  # astype() equivalent to floor() + casting
        expert = np.maximum(0, expert)
        expert = np.minimum(self.num_experts - 1, expert)

        out = np.multiply(keys, self.stage_2_w[expert])
        out = np.add(out, self.stage_2_b[expert])

        return (out, expert)


class IndexStructureGapped(object):
    """Defines class IndexStructureGapped, which implements Select, Insert,
       and Delete functionality using a numpy array for storage. Gaps
       are left in the array to allow for fast insert operations.
    """

    def __init__(self, model, scale):
        """Initialize class

        Args:
            model: An instance of class RMIsimple
            scale: Integer, indicates size of gapped array relative to key array.


        Returns:
            -
        """

        self._model = model
        self._scale = scale

        # Makes a copy of the data_set, and converts to numpy array.
        keys = np.array(model._data_set.keys)

        if len(keys.shape) > 1:
            if keys.shape[1] > 1:
                raise ValueError("Key_size must be 1")
            else:
                keys = keys.squeeze()

        # Sort the keys, for fast Select, Insert, and Delete
        self._keys_array = np.sort(np.array(keys))

        # Put gaps into the array
        self._is_key = np.full(self._keys_array.shape, True)
        self.rescale()

    def rescale(self, scale=None):
        """Rescales the size of the array, adding gaps between keys,
           with the new array being scale times larger.

        Args:
            scale: Integer. New array will be scale times larger.
                   Note that scale must be greater or equal to 1.

        Returns:
            -
            (modifies self.keys_array and self.is_key)
        """

        if scale is not None:
            self._scale = scale

        # First remove gaps from array
        self._keys_array = self._keys_array[self._is_key]

        num_keys = self._keys_array.shape[0]

        # Construct new array with gaps, using is_key to keep track
        # of which elements are keys.

        trues = np.full(num_keys, True)
        falses = np.full(num_keys, False)
        keys = self._keys_array

        # Each key should repeat scale number of times
        self._keys_array = np.repeat(keys, self._scale)

        # Initialize _is_key array to False
        self._is_key = np.full(self._scale * num_keys,
                               False)
        # Then mark as true the first instance of each key
        self._is_key[0::self._scale] = True

    def _find_insert_position(self, keys):
        """Helper function that runs inference to find position
           for Select, Insert, and Delete.

        Args:
            keys: numpy array of keys

        Returns:
            positions: Array of positions. For each key, the returned position
                is leftmost position in the sorted array where Insert would
                result in a sorted array.
        """

        num_keys = len(keys)
        keys = np.reshape(keys, (num_keys, 1))

        # Run inference
        model_positions, experts = self._model.run_inference(keys)
        model_positions = np.reshape(model_positions, (num_keys,))
        experts = np.reshape(experts, (num_keys,))

        # Inference prediction is typically not correct position,
        # so we need to conduct a binary search based on known
        # maximum inference error for each expert

        pos_output = np.zeros(num_keys, dtype=np.int64)
        for idx in range(num_keys):

            expert = experts[idx]
            model_pos = np.round(model_positions[idx]).astype(np.int64)

            max_error_left = int(self._model.max_error_left[expert])
            max_error_right = int(self._model.max_error_right[expert])

            min_pos = self._model.min_pos[expert]
            max_pos = self._model.max_pos[expert]

            max_key_idx = self._keys_array.shape[0] - 1

            # Leftmost search pos should typically be (model_pos - max_error_left),
            # but must also lie between 0 and max_key_idx
            search_range_left = model_pos - max_error_left
            search_range_left = np.maximum(0,
                                           search_range_left)
            search_range_left = np.minimum(max_key_idx,
                                           search_range_left)

            # Rightmost search pos should typically be (model_pos + max_error_right),
            # but must also lie between 0 and max_key_idx
            search_range_right = model_pos + max_error_right
            search_range_right = np.maximum(0,
                                            search_range_right)
            search_range_right = np.minimum(max_key_idx,
                                            search_range_right)

            search_range = [search_range_left, search_range_right]

            # Before conducting the search, check whether the error bounds are large enough
            leftmost_key = self._keys_array[search_range[0]]
            rightmost_key = self._keys_array[search_range[1]]

            if leftmost_key <= keys[idx] <= rightmost_key:
                # If the key lies within the range, search for it with binary search
                found_pos = np.searchsorted(self._keys_array[search_range[0]:search_range[1] + 1],
                                            keys[idx],
                                            side='left')
                # Because np.searchsorted returns an array with one element:
                found_pos = found_pos[0]
                # Adjust found_pos for full keys_array, not just for the slice
                found_pos += search_range[0]

            elif leftmost_key > keys[idx]:
                # If the key lies to the left of the range, just scan to the left incrementally
                pos = search_range[0] - 1
                while pos >= 0:
                    if self._keys_array[pos] < keys[idx]:
                        found_pos = pos + 1
                        break
                    pos -= 1
                if pos == -1:
                    found_pos = 0

            elif rightmost_key < keys[idx]:
                # If the key lies to the right of the range, just scan to the right incrementally
                pos = search_range[1] + 1
                while pos <= self._keys_array.shape[0] - 1:
                    if self._keys_array[pos] >= keys[idx]:
                        found_pos = pos
                        break
                    pos += 1
                if pos == self._keys_array.shape[0]:
                    found_pos = pos

            pos_output[idx] = found_pos

        return pos_output

    def select(self, keys):
        """Return position(s) of key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            positions: Numpy array positions of the keys.
                If a key is not found, the returned position is -1.
        """
        pos_candidates = self._find_insert_position(keys)

        num_keys = len(keys)

        pos_output = np.zeros(num_keys, dtype=np.int64)

        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]

            if (pos < self._keys_array.shape[0]
                    and self._keys_array[pos] == key
                    and self._is_key[pos] == True):

                pos_output[idx] = pos_candidates[idx]
            else:
                pos_output[idx] = -1

        return pos_output

    def insert(self, keys):
        """Insert key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            success: Numpy boolean array: True for each successful insertion;
                False for each failed insertion (due to key already in array).
        """

        success = np.full(len(keys), False)

        for idx, key in enumerate(keys):

            pos = self._find_insert_position([key])
            pos = pos[0]

            if (pos < self._keys_array.shape[0]
                    and self._keys_array[pos] == key
                    and self._is_key[pos] == True):
                # If the key already exists, no insertation takes place
                # and the output is success=False
                success[idx] = False
            else:
                # If the key did not already exist, then output success=True.
                success[idx] = True

            # Don't procede with insert if key already exists
            if not success[idx]:
                continue

            # Search to the left and right until first available position
            # is found.

            left_pos = pos - 1
            right_pos = pos

            # Check whether there is space to the left or right to search
            if left_pos < 0:
                more_left = False
            else:
                more_left = True
            if right_pos >= self._keys_array.shape[0]:
                more_right = False
            else:
                more_right = True

            # Keep search while there is room to the left or right
            while more_left or more_right:

                if more_right:
                    if self._is_key[right_pos] == False:
                        # If a gap is found, shift the data around and fill in the gap
                        self._is_key[right_pos] = True
                        self._keys_array[pos + 1:right_pos + 1] = self._keys_array[pos:right_pos]
                        self._keys_array[pos] = key
                        position_range_for_errors_update = [pos, right_pos + 1]
                        break
                    else:
                        # If no gap, increment to the right
                        right_pos += 1
                        if right_pos >= self._keys_array.shape[0]:
                            more_right = False

                if more_left:
                    if self._is_key[left_pos] == False:
                        # If a gap is found, shift the data around and fill in the gap
                        self._is_key[left_pos] = True
                        self._keys_array[left_pos:pos - 1] = self._keys_array[left_pos + 1:pos]
                        self._keys_array[pos - 1] = key
                        position_range_for_errors_update = [left_pos, pos]
                        break
                    else:
                        # If no gap, increment to the left
                        left_pos -= 1
                        if left_pos < 0:
                            more_left = False

            # If the above loop terminates without finding a gap...
            if more_left == False and more_right == False:
                # No insertaion position found
                print("Warning: no gaps left to insert key.")
                success[idx] = False

            # Update errors for all keys that have moved
            pos_range = position_range_for_errors_update
            key_pos = []
            for pos in range(pos_range[0], pos_range[1]):
                if self._is_key[pos]:
                    key_pos.append([[self._keys_array[pos]],
                                    pos])
            self._model.calc_min_max_errors(key_pos)

        return success

    def delete(self, keys):
        """Delete key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            success: Numpy boolean array: True for each successful deletion;
                False for each failed deletion (due to key not in array).
        """

        pos_candidates = self._find_insert_position(keys)

        num_keys = len(keys)

        success = np.full(num_keys, False)

        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]

            if (pos < self._keys_array.shape[0]
                    and self._keys_array[pos] == key
                    and self._is_key[pos]):
                # If the key already exists, deletion can take place,
                # so the output is success=False.
                success[idx] = True
            else:
                # If the key did not already exist, then output success=False.
                success[idx] = False

        # self._keys_array = np.delete(self._keys_array, pos_candidates[success])

        self._is_key[pos_candidates[success]] = False

        return success

    def train(self,
              batch_sizes=None,
              max_steps=None,
              learning_rates=None,
              model_save_dir=None):
        """Train the model, calculate expert errors, etc. Fully prepares
           the model for Select, Insert, and Delete operations.
           This function should be used after significant number of
           insertions and deletions.

        Args:
            batch_sizes: list (length=2) of batch sizes for the two Stages.
                default=None (use the model's self.batch_sizes)
            max_steps: list (length=2) of number of training steps for the two Stages.
                default=None (use the model's self.max_steps)
            learning_rates: list (length=2) of learning rates for the two Stages.
                default=None (use the model's self.learning_rates)
            model_save_dir: Name of directory to save the model
                default=None (use the model's self.model_save_dir)

        Returns:
            -
        """

        # Construct new DataSet from current array of keys,
        # to be used to retrain the model.
        # Assumes that self._keys_array is already sorted.

        new_keys = self._keys_array[self._is_key]
        num_keys = new_keys.shape[0]
        new_key_positions = np.arange(self._keys_array.shape[0])[self._is_key]
        data_set = DataSet(keys=np.reshape(new_keys, [-1, 1]),
                           positions=new_key_positions,
                           num_positions=self._keys_array.shape[0])

        self._model.new_data(data_set)

        # Train on the new data set.
        self._model.run_training(batch_sizes=batch_sizes,
                                 max_steps=max_steps,
                                 learning_rates=learning_rates,
                                 model_save_dir=model_save_dir)

        # Prepare weights for fast inference
        self._model.get_weights_from_trained_model()

        # Calculate the Stage 2 errors
        self._model.calc_min_max_errors()


class IndexStructurePacked(object):
    """Defines class IndexStructurePacked, which implements Select, Insert,
       and Delete functionality using a packed numpy array for storage.
       This choice does not allow efficient Insert and Delete, but is
       straightforward to implement.

       The plan is to implement a new class, IndexStructureGapped, that will
       replace IndexStructurePacked in a future revision. The Gapped version
       will leaves gaps in the array, allowing for faster Insert.

    """

    def __init__(self, model):
        """Initialize class

        Args:
            model: An instance of class RMIsimple

        Returns:
            -
        """

        self._model = model

        # Makes a copy of the data_set, and converts to numpy array.
        keys = np.array(model._data_set.keys)

        if len(keys.shape) > 1:
            if keys.shape[1] > 1:
                raise ValueError("Key_size must be 1")
            else:
                keys = keys.squeeze()

        # Sort the keys, for fast Select, Insert, and Delete
        self._keys_array = np.sort(np.array(keys))

    def _find_insert_position(self, keys):
        """Helper function that runs inference to find position
           for Select, Insert, and Delete.

        Args:
            keys: numpy array of keys

        Returns:
            positions: Array of positions. For each key, the returned position
                is leftmost position in the sorted array where Insert would
                result in a sorted array.
        """

        num_keys = len(keys)
        keys = np.reshape(keys, (num_keys, 1))

        # Run inference
        model_positions, experts = self._model.run_inference(keys)
        model_positions = np.reshape(model_positions, (num_keys,))
        experts = np.reshape(experts, (num_keys,))

        # Inference prediction is typically not correct position,
        # so we need to conduct a binary search based on known
        # maximum inference error for each expert

        pos_output = np.zeros(num_keys, dtype=np.int64)
        for idx in range(num_keys):

            expert = experts[idx]
            model_pos = np.round(model_positions[idx]).astype(np.int64)

            max_error_left = int(self._model.max_error_left[expert])
            max_error_right = int(self._model.max_error_right[expert])

            min_pos = self._model.min_pos[expert]
            max_pos = self._model.max_pos[expert]

            max_key_idx = self._keys_array.shape[0] - 1

            # Leftmost search pos should typically be (model_pos - max_error_left),
            # but must also lie between 0 and (max_key_idx - max_error_left)
            search_range_left = model_pos - max_error_left
            search_range_left = np.maximum(0,
                                           search_range_left)
            search_range_left = np.minimum(max_key_idx - max_error_left,
                                           search_range_left)

            # Rightmost search pos should typically be (model_pos + max_error_right),
            # but must also lie between max_error_right and max_key_idx
            search_range_right = model_pos + max_error_right
            search_range_right = np.maximum(max_error_right,
                                            search_range_right)
            search_range_right = np.minimum(max_key_idx,
                                            search_range_right)

            search_range = [search_range_left, search_range_right]

            # Before conducting the search, check whether the error bounds are large enough
            leftmost_key = self._keys_array[search_range[0]]
            rightmost_key = self._keys_array[search_range[1]]

            if leftmost_key <= keys[idx] <= rightmost_key:
                # If the key lies within the range, search for it with binary search
                found_pos = np.searchsorted(self._keys_array[search_range[0]:search_range[1] + 1],
                                            keys[idx],
                                            side='left')
                # Because np.searchsorted returns an array with one element:
                found_pos = found_pos[0]
                # Adjust found_pos for full keys_array, not just for the slice
                found_pos += search_range[0]

            elif leftmost_key > keys[idx]:
                # If the key lies to the left of the range, just scan to the left incrementally
                pos = search_range[0] - 1
                while pos >= 0:
                    if self._keys_array[pos] < keys[idx]:
                        found_pos = pos + 1
                        break
                    pos -= 1
                if pos == -1:
                    found_pos = 0

            elif rightmost_key < keys[idx]:
                # If the key lies to the right of the range, just scan to the right incrementally
                pos = search_range[1] + 1
                while pos <= self._keys_array.shape[0] - 1:
                    if self._keys_array[pos] >= keys[idx]:
                        found_pos = pos
                        break
                    pos += 1
                if pos == self._keys_array.shape[0]:
                    found_pos = pos

            pos_output[idx] = found_pos

        return pos_output

    def select(self, keys):
        """Return position(s) of key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            positions: Numpy array positions of the keys.
                If a key is not found, its position is set to -1.
        """
        pos_candidates = self._find_insert_position(keys)

        num_keys = len(keys)

        pos_output = np.zeros(num_keys, dtype=np.int64)

        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]

            if (pos < self._keys_array.shape[0]
                    and self._keys_array[pos] == key):

                pos_output[idx] = pos_candidates[idx]
            else:
                pos_output[idx] = -1

        return pos_output

    def insert(self, keys):
        """Insert key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            success: Numpy boolean array: True for each successful insertion;
                False for each failed insertion (due to key already in array).
        """

        pos_candidates = self._find_insert_position(keys)

        num_keys = len(keys)

        success = np.full(num_keys, False)

        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]

            if (pos < self._keys_array.shape[0]
                    and self._keys_array[pos] == key):
                # If the key already exists, no insertation takes place
                # and the output is success=False
                success[idx] = False
            else:
                # If the key did not already exist, then output success=True.
                success[idx] = True

        # Only work with keys that are to be inserted
        keys = np.array(keys)[success]
        pos = np.array(pos_candidates)[success]

        # When inserting multiple keys, divide the keys into three groups:
        # keys_before: keys to be inserted to left of self._keys_array
        # keys_middle: keys to be inserted within the array
        # keys_after: keys to be inserted to right of self._keys_array

        # Sort by key
        perm = np.argsort(keys)
        keys = keys[perm]
        pos = pos[perm]

        keys_array_size = self._keys_array.shape[0]

        left_idx = np.searchsorted(pos, 0, side='right')
        right_idx = np.searchsorted(pos, keys_array_size, side='left')

        # keys_before = keys[:left_idx]
        # keys_middle = keys[left_idx:right_idx]
        # keys_after = keys[right_idx:]

        # Alternatively, use np.split to get list of three arrays
        # described above.
        keys_split = np.split(keys, [left_idx, right_idx])
        pos_split = np.split(pos, [left_idx, right_idx])

        # Insert into the middle first:
        # Newer version of numpy allow multiple inserts
        self._keys_array = np.insert(self._keys_array,
                                     pos_split[1],
                                     keys_split[1])

        # Now concatenate with keys_before and keys_after
        self._keys_array = np.concatenate((keys_split[0],
                                           self._keys_array,
                                           keys_split[2]))

        return success

    def delete(self, keys):
        """Delete key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            success: Numpy boolean array: True for each successful deletion;
                False for each failed deletion (due to key not in array).
        """

        pos_candidates = self._find_insert_position(keys)

        num_keys = len(keys)

        success = np.full(num_keys, False)

        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]

            if (pos < self._keys_array.shape[0]
                    and self._keys_array[pos] == key):
                # If the key already exists, deletion can take place,
                # so the output is success=False.
                success[idx] = True
            else:
                # If the key did not already exist, then output success=False.
                success[idx] = False

        self._keys_array = np.delete(self._keys_array, pos_candidates[success])

        return success

    def train(self,
              batch_sizes=None,
              max_steps=None,
              learning_rates=None,
              model_save_dir=None):
        """Train the model, calculate expert errors, etc. Fully prepares
           the model for Select, Insert, and Delete operations.
           This function should be used after significant number of
           insertions and deletions.

        Args:
            batch_sizes: list (length=2) of batch sizes for the two Stages.
                default=None (use the model's  self.batch_sizes)
            max_steps: list (length=2) of number of training steps for the two Stages.
                default=None (use the model's self.max_steps)
            learning_rates: list (length=2) of learning rates for the two Stages.
                default=None (use the model's self.learning_rates)
            model_save_dir: Name of directory to save the model
                default=None (use the model's self.model_save_dir)

        Returns:
            -
        """

        # Construct new DataSet from current array of keys,
        # to be used to retrain the model.
        # Assumes that self._keys_array is already sorted.

        num_keys = self._keys_array.shape[0]
        data_set = DataSet(np.reshape(self._keys_array, [-1, 1]), np.arange(num_keys))

        self._model.new_data(data_set)

        # Train on the new data set.
        self._model.run_training(batch_sizes=batch_sizes,
                                 max_steps=max_steps,
                                 learning_rates=learning_rates,
                                 model_save_dir=model_save_dir)

        # Prepare weights for fast inference
        self._model.get_weights_from_trained_model()

        # Calculate the Stage 2 errors
        self._model.calc_min_max_errors()


def save_db(db_obj, f_name):
    """Save the database.
    Args:
        db_obj: Instance of type IndexStructurePacked or IndexStructureGapped.

        f_name: File name (string) of save file.

    Returns:
        -
    """
    with open(f_name, 'wb') as file:
        pickle.dump(db_obj, file)


def load_db(f_name):
    """Load a database.
    Args:
        f_name: File name (string) of database previously saved with save_db().

    Returns:

        The saved database instance.

    """
    with open(f_name, 'rb') as file:

        return pickle.load(file)


def own_dataset(key):
    num_keys = len(key)
    keys = np.array(key)
    positions = np.arange(num_keys)
    raw_data_set = DataSet(keys=keys, positions=positions)
    # Split into train/validate, using 100% for training (no validation needed)

    data_sets = create_train_validate_data_sets(raw_data_set, validation_size=0)
    return data_sets


def train(data_sets, mode_save):
    # Create a Recursive-model index based on the training data set
    # mode_save: 'learned_index'

    rmi = RmiSimple(data_sets.train, hidden_layer_widths=[8, 8], num_experts=100)

    # Create a learned index structure, which can be used like a database.
    # Choose either IndexStructurePacked or IndexStructure_Gapped.

    # IndexStructure_Gapped is faster for insertions and deletions.

    rmi_db = IndexStructureGapped(model=rmi, scale=2)  # scale

    # If using IndexStructure_Gapped, you can rescale the array at any time.
    rmi_db.rescale(scale=2)

    # IndexStructurePacked uses less space.
    # Comment the above code and uncomment the following code if you
    # want to use IndexStructurePacked instead.
    # rmi_db = li.database.IndexStructurePacked(model=rmi)

    # Train the database

    # May need to try different batch_sizes, max_steps, learning rates.
    # Each is an array with two elements (for Stage 1 and Stage 2).

    # Note that rmi_db.train() not only trains the model, but also
    # calculates and saves the maximum errors for each "expert" and
    # saves the trained weights and biases for use in fast Numpy
    # inference calculations. Basically, this function does everything
    # needed to get Select, Insert, and Delete ready to work.

    # batch_sizes=[10000, 1000]

    rmi_db.train(batch_sizes=[512, 8],
                 max_steps=[500, 500],
                 learning_rates=[0.001, 1000],
                 model_save_dir=mode_save)

    # batch_sizes are the batch sizes for the two stages.
    # max_steps are the maximum number of batches for each stage.
    # learning_rates are the learning rates for the two stages.
    # model_save_dir is where to save the trained model.

    save_db(rmi_db, mode_save + str('/temp_db.p'))


class UseLearnedIndex(object):
    def __init__(self, config):
        self.config = config
        self.all_segment_path = np.load(self.config.data_dir + '/segment_all_dict.npy', allow_pickle=True)
        self.all_segment = self.all_segment_path.item()
        self.original_segment_path = np.load(self.config.data_dir + '/original_segment.npy', allow_pickle=True)
        self.original_segment = self.original_segment_path.item()
        self.learned_index_path = self.config.data_dir + '/learned_index'
        self.all_key = np.load(self.config.data_dir + '/hexagon/all_key.npy', allow_pickle=True)
        self.side_length = 500
        self.big_hexagon_path = np.load(self.config.data_dir + '/hexagon/big_hexagon.npy', allow_pickle=True)
        self.big_hexagon = self.big_hexagon_path.item()
        self.c = [10 ** (3 + len(str(self.side_length))), 10 ** len(str(self.side_length)), 1]
        self.each_degree = 60

        self.key_dict_path = np.load(self.config.data_dir + '/hexagon/key_dict.npy', allow_pickle=True)
        self.key_dict = self.key_dict_path.item()

        self.count_fractal_hexagon_path = np.load(self.config.data_dir + '/hexagon/count_fractal_big_hexagon.npy', allow_pickle=True)
        self.count_fractal_hexagon = []
        for item in self.count_fractal_hexagon_path:
            self.count_fractal_hexagon.append(item)

        self.get_small_hexagon_path = np.load(self.config.data_dir + '/hexagon/get_small_hexagon.npy', allow_pickle=True)
        self.get_small_hexagon = self.get_small_hexagon_path.item()

        self.construct()
        self.key_model = load_db(self.learned_index_path + '/temp_db.p')

    def construct(self):
        if not os.path.exists(self.learned_index_path):
            os.makedirs(self.learned_index_path)
            key_dataset = own_dataset(self.all_key)
            train(key_dataset, self.learned_index_path)

    def search(self, point, radius):
        side_length_dis = self.side_length * math.sqrt(3) / 2
        small_side_length = self.side_length / math.sqrt(7)
        small_side_length_dis = self.side_length / math.sqrt(7) * math.sqrt(3) / 2

        start_time = time.time()
        cell_id = label_point(point, self.side_length, self.big_hexagon)
        cell_center_point = self.big_hexagon[cell_id]['center_point']
        big_degree = get_bear(cell_center_point, point)
        big_meter = round(distance_meter(cell_center_point, point), 0)
        # print('big_meter', big_meter)
        big_meter_plus_radius = big_meter + radius
        big_meter_sub_radius = big_meter - radius

        s_id = []
        key_range = []
        if cell_id in self.count_fractal_hexagon:
            small_label = math.ceil(
                (((big_degree + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)
            small_point = self.get_small_hexagon[round(cell_id + 0.1 * small_label, 1)]['center_point']
            to_small_meter = distance_meter(small_point, point)

            if big_meter <= to_small_meter:
                label = cell_id
                if big_meter <= radius:
                    for i in range(math.ceil(360 / self.each_degree) + 1):
                        key_min = self.c[0] * label + self.c[1] * i
                        key_max = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                else:
                    degree_range = round(math.degrees(math.asin(radius / big_meter)), 5)
                    degree_min = big_degree - degree_range
                    degree_max = big_degree + degree_range
                    degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                    degree_max_block = math.ceil((degree_max % 360) / self.each_degree)

                    if degree_min_block == degree_max_block:
                        key_min = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                    elif degree_max_block > degree_min_block:
                        for i in range(degree_max_block - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                            key_range += [key_min, key_max]

                    else:
                        for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                            key_range += [key_min, key_max]

                        for i in range(degree_max_block):
                            key_min = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * big_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * big_meter_plus_radius
                            key_range += [key_min, key_max]

                    if big_meter + radius > small_side_length_dis:
                        small_label1 = small_label
                        small_point1 = small_point
                        small_new_degree1 = get_bear(small_point1, point)
                        dis_small_point1 = distance_meter(small_point1, point)
                        to_small_meter1 = dis_small_point1 - radius
                        dis_small_point1_plus_radius = dis_small_point1 + radius

                        if to_small_meter1 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter1 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                            degree_min = small_new_degree1 - degree_range
                            degree_max = small_new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                key_max = self.c[0] * small_label1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + \
                                              self.c[2] * to_small_meter1

                                    key_max = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + \
                                              self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1
                                    key_max = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label1 + self.c[1] * i + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                        small_label2_text = (small_label1 + 1 - 2 * (small_label1 - round(
                            (((big_degree + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60,
                            0))) % 6

                        small_label2 = 6 if small_label2_text == 0 else small_label2_text

                        small_point2 = self.get_small_hexagon[round(cell_id + 0.1 * small_label2, 1)]['center_point']
                        small_new_degree2 = get_bear(small_point2, point)
                        dis_small_point2 = distance_meter(small_point2, point)
                        to_small_meter2 = dis_small_point2 - radius
                        dis_small_point2_plus_radius = dis_small_point2 + radius

                        if to_small_meter2 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter2 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                            degree_min = small_new_degree2 - degree_range
                            degree_max = small_new_degree2 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                key_max = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2

                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

            else:
                label = round(cell_id + 0.1 * small_label, 1)
                # print('new_label', label)
                to_small_degree = get_bear(small_point, point)
                to_small_meter_sub_radius = to_small_meter - radius
                to_small_meter_plus_radius = to_small_meter + radius

                if to_small_meter <= radius:
                    for i in range(math.ceil(360 / self.each_degree) + 1):
                        key_min = self.c[0] * label + self.c[1] * i
                        key_max = self.c[0] * label + self.c[1] * i + self.c[2] * to_small_meter_plus_radius
                        key_range += [key_min, key_max]

                else:
                    degree_range = round(math.degrees(math.asin(radius / to_small_meter)), 5)
                    degree_min = to_small_degree - degree_range
                    degree_max = to_small_degree + degree_range
                    degree_min_block = math.ceil(
                        ((degree_min + 30 - math.degrees(
                            math.asin((1 / 2) / math.sqrt(7)))) % 360) / self.each_degree)
                    degree_max_block = math.ceil(
                        ((degree_max + 30 - math.degrees(
                            math.asin((1 / 2) / math.sqrt(7)))) % 360) / self.each_degree)

                    if degree_min_block == degree_max_block:
                        key_min = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * to_small_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * to_small_meter_plus_radius
                        key_range += [key_min, key_max]

                    elif degree_max_block > degree_min_block:
                        for i in range(degree_max_block - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_plus_radius
                            key_range += [key_min, key_max]
                    else:
                        for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_plus_radius
                            key_range += [key_min, key_max]

                        for i in range(degree_max_block):
                            key_min = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * to_small_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * to_small_meter_plus_radius
                            key_range += [key_min, key_max]

                    if big_meter + radius > side_length_dis:
                        new_center_degree1 = (big_degree // 60) * 60
                        new_center_point1 = destination(cell_center_point, new_center_degree1,
                                                        dist=math.sqrt(3) * self.side_length)
                        new_center_label1 = label_point(new_center_point1, self.side_length, self.big_hexagon)
                        new_degree1 = get_bear(new_center_point1, point)

                        if new_center_label1 in self.count_fractal_hexagon:
                            small_label_1 = math.ceil(
                                (((new_degree1 + 60 - math.degrees(
                                    math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                            small_point1 = self.get_small_hexagon[round(new_center_label1 + 0.1 * small_label_1, 1)][
                                'center_point']
                            dis_small_point1 = distance_meter(small_point1, point)
                            to_small_meter1 = dis_small_point1 - radius
                            dis_small_point1_plus_radius = dis_small_point1 + radius
                            if to_small_meter1 < 0:
                                for i in range(math.ceil(360 / self.each_degree) + 1):
                                    key_min = self.c[0] * label + self.c[1] * i
                                    key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            elif to_small_meter1 < small_side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                                degree_min = new_degree1 - degree_range
                                degree_max = new_degree1 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter1

                                        key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter1
                                        key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * to_small_meter1
                                        key_max = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                        key_range += [key_min, key_max]

                        else:
                            dis_new_center_point1 = distance_meter(point, new_center_point1)
                            new_label1_meter = dis_new_center_point1 - radius
                            dis_new_center_point1_plus_radius = dis_new_center_point1 + radius

                            if new_label1_meter < self.side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_new_center_point1)), 5)
                                degree_min = new_degree1 - degree_range
                                degree_max = new_degree1 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[
                                        2] * new_label1_meter
                                    key_max = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label1_meter

                                        key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label1_meter
                                        key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * new_label1_meter
                                        key_max = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * dis_new_center_point1_plus_radius
                                        key_range += [key_min, key_max]

                        new_center_degree2 = (big_degree // 60 + 1) * 60
                        new_center_point2 = destination(cell_center_point, new_center_degree2,
                                                        dist=math.sqrt(3) * self.side_length)
                        new_center_label2 = label_point(new_center_point2, self.side_length, self.big_hexagon)
                        new_degree2 = get_bear(new_center_point2, point)

                        if new_center_label2 in self.count_fractal_hexagon:
                            small_label_2 = math.ceil(
                                (((new_degree2 + 60 - math.degrees(
                                    math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                            small_point2 = self.get_small_hexagon[round(new_center_label2 + 0.1 * small_label_2, 1)][
                                'center_point']
                            dis_small_point2 = distance_meter(small_point2, point)
                            to_small_meter2 = dis_small_point2 - radius
                            dis_small_point2_plus_radius = dis_small_point2 + radius
                            if to_small_meter2 < 0:
                                for i in range(math.ceil(360 / self.each_degree) + 1):
                                    key_min = self.c[0] * label + self.c[1] * i
                                    key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            elif to_small_meter2 < small_side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                                degree_min = new_degree1 - degree_range
                                degree_max = new_degree1 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter2

                                        key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter2
                                        key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * to_small_meter2
                                        key_max = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                        key_range += [key_min, key_max]
                        else:
                            dis_new_center_point2 = distance_meter(point, new_center_point2)
                            new_label2_meter = dis_new_center_point2 - radius
                            dis_new_center_point2_plus_radius = dis_new_center_point2 + radius
                            if new_label2_meter < self.side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_new_center_point2)), 5)
                                degree_min = new_degree2 - degree_range
                                degree_max = new_degree2 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[
                                        2] * new_label2_meter
                                    key_max = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label2_meter

                                        key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label2_meter
                                        key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * new_label2_meter
                                        key_max = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * dis_new_center_point2_plus_radius
                                        key_range += [key_min, key_max]

                    elif to_small_meter + radius > small_side_length_dis:
                        small_label_1 = label

                        small_point1 = cell_center_point

                        new_degree1 = get_bear(small_point1, point)

                        dis_small_point1 = distance_meter(small_point1, point)
                        to_small_meter1 = dis_small_point1 - radius
                        dis_small_point1_plus_radius = dis_small_point1 + radius

                        if to_small_meter1 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter1 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                key_max = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1

                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                        small_label2_text = (small_label + 1 - 2 * (small_label - round(
                            (((big_degree + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60,
                            0))) % 6
                        small_label2 = 6 if small_label2_text == 0 else small_label2_text
                        small_point2 = self.get_small_hexagon[round(cell_id + 0.1 * small_label2, 1)]['center_point']
                        small_new_degree2 = get_bear(small_point2, point)
                        dis_small_point2 = distance_meter(small_point2, point)
                        to_small_meter2 = dis_small_point2 - radius
                        dis_small_point2_plus_radius = dis_small_point2 + radius
                        if to_small_meter2 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter2 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                            degree_min = small_new_degree2 - degree_range
                            degree_max = small_new_degree2 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                key_max = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2

                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

        else:
            label = cell_id
            if big_meter <= radius:
                for i in range(math.ceil(360 / self.each_degree) + 1):
                    key_min = self.c[0] * label + self.c[1] * i
                    key_max = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_plus_radius
                    key_range += [key_min, key_max]

            else:
                degree_range = round(math.degrees(math.asin(radius / big_meter)), 5)
                degree_min = big_degree - degree_range
                degree_max = big_degree + degree_range
                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                if degree_min_block == degree_max_block:
                    key_min = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_sub_radius
                    key_max = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_plus_radius
                    key_range += [key_min, key_max]

                elif degree_max_block > degree_min_block:
                    for i in range(degree_max_block - degree_min_block + 1):
                        key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                else:
                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                        key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                    for i in range(degree_max_block + 1):
                        key_min = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                if big_meter + radius > side_length_dis:
                    new_center_degree1 = (big_degree // 60) * 60
                    new_center_point1 = destination(cell_center_point, new_center_degree1,
                                                    dist=math.sqrt(3) * self.side_length)
                    new_center_label1 = label_point(new_center_point1, self.side_length, self.big_hexagon)
                    new_degree1 = get_bear(new_center_point1, point)

                    if new_center_label1 in self.count_fractal_hexagon:
                        small_label_1 = math.ceil(
                            (((new_degree1 + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                        small_point1 = self.get_small_hexagon[round(new_center_label1 + 0.1 * small_label_1, 1)][
                            'center_point']
                        dis_small_point1 = distance_meter(small_point1, point)
                        to_small_meter1 = dis_small_point1 - radius
                        dis_small_point1_plus_radius = dis_small_point1 + radius

                        if to_small_meter1 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter1 < small_side_length:
                            # print('dis_small_point1', radius, dis_small_point1)
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                key_max = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter1

                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                    else:
                        dis_new_center_point1 = distance_meter(point, new_center_point1)
                        new_label1_meter = dis_new_center_point1 - radius
                        dis_new_center_point1_plus_radius = dis_new_center_point1 + radius

                        if new_label1_meter < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label +self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_new_center_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif new_label1_meter < self.side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_new_center_point1)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[2] * new_label1_meter
                                key_max = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label1_meter

                                    key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label1_meter
                                    key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * new_label1_meter
                                    key_max = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                    new_center_degree2 = (big_degree // 60 + 1) * 60
                    new_center_point2 = destination(cell_center_point, new_center_degree2, math.sqrt(3) * self.side_length)
                    new_center_label2 = label_point(new_center_point2, self.side_length, self.big_hexagon)
                    new_degree2 = get_bear(new_center_point2, point)

                    if new_center_label2 in self.count_fractal_hexagon:
                        small_label_2 = math.ceil(
                            (((new_degree2 + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                        small_point2 = self.get_small_hexagon[round(new_center_label2 + 0.1 * small_label_2, 1)][
                            'center_point']
                        dis_small_point2 = distance_meter(small_point2, point)
                        to_small_meter2 = dis_small_point2 - radius
                        dis_small_point2_plus_radius = dis_small_point2 + radius

                        if to_small_meter2 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter2 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                key_max = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2

                                    key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2
                                    key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]
                    else:
                        dis_new_center_point2 = distance_meter(new_center_point2, point)
                        new_label2_meter = dis_new_center_point2 - radius
                        dis_new_center_point2_plus_radius = dis_new_center_point2 + radius

                        if new_label2_meter < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_new_center_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif new_label2_meter < self.side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_new_center_point2)), 5)
                            degree_min = new_degree2 - degree_range
                            degree_max = new_degree2 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[2] * new_label2_meter
                                key_max = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label2_meter

                                    key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label2_meter
                                    key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]
                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * new_label2_meter
                                    key_max = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]

        pos = self.key_model._find_insert_position(key_range)

        for i in range(int(len(pos)/2)):
            pos_min = pos[i * 2]
            pos_max = pos[i * 2 + 1]
            if pos_min == pos_max and pos_min % 2 != 0:
                pass
            else:
                idx1 = pos_min if pos_min % 2 == 0 else pos_min + 1
                idx2 = pos_max if pos_max % 2 == 0 else pos_max - 1

                s_id += [self.key_dict[int(idx1) + j * 2]['link_id'] for j in range(int((idx2 - idx1) / 2) + 1)]

        new_s_id = [link.split('_')[0] for link in s_id]
        new_s_id = list(np.unique(new_s_id))
        candidate_segment = []

        for link in new_s_id:
            link_start = self.original_segment[link]['node_a_id']
            link_end = self.original_segment[link]['node_b_id']
            d, p, _ = distance_point_to_segment(point, self.original_segment[link]['node_a'],
                                                self.original_segment[link]['node_b'])
            if d <= radius:
                candidate_segment.append([link, link_start, link_end, p, d])
        candidate_segment.sort(key=lambda x: x[-1])
        candidate_time = round(time.time() - start_time, 9)
        # print(len(candidate_segment))
        return candidate_segment, candidate_time


class UseLearnedIndex_param(object):
    def __init__(self, config,side_length, each_degree, hexagon_name):
        self.config = config
        self.hexagon_name = hexagon_name
        self.all_segment_path = np.load(self.config.data_dir + '/' + self.hexagon_name + '/segment_all_dict.npy', allow_pickle=True)
        self.all_segment = self.all_segment_path.item()
        self.original_segment_path = np.load(self.config.data_dir + '/' + self.hexagon_name + '/original_segment.npy', allow_pickle=True)
        self.original_segment = self.original_segment_path.item()
        self.learned_index_path = self.config.data_dir + '/learned_index' + self.hexagon_name
        self.all_key = np.load(self.config.data_dir + '/' + self.hexagon_name + '/all_key.npy', allow_pickle=True)
        self.side_length = side_length
        self.big_hexagon_path = np.load(self.config.data_dir + '/' + self.hexagon_name + '/big_hexagon.npy', allow_pickle=True)
        self.big_hexagon = self.big_hexagon_path.item()
        self.c = [10 ** (3 + len(str(self.side_length))), 10 ** len(str(self.side_length)), 1]
        self.each_degree = each_degree

        self.key_dict_path = np.load(self.config.data_dir + '/' + self.hexagon_name + '/key_dict.npy', allow_pickle=True)
        self.key_dict = self.key_dict_path.item()

        self.count_fractal_hexagon_path = np.load(self.config.data_dir + '/' + self.hexagon_name + '/count_fractal_big_hexagon.npy', allow_pickle=True)
        self.count_fractal_hexagon = []
        for item in self.count_fractal_hexagon_path:
            self.count_fractal_hexagon.append(item)

        self.get_small_hexagon_path = np.load(self.config.data_dir + '/' + self.hexagon_name + '/get_small_hexagon.npy', allow_pickle=True)
        self.get_small_hexagon = self.get_small_hexagon_path.item()

        self.construct()
        self.key_model = load_db(self.learned_index_path + '/temp_db.p')

    def construct(self):
        if not os.path.exists(self.learned_index_path):
            os.makedirs(self.learned_index_path)
            key_dataset = own_dataset(self.all_key)
            train(key_dataset, self.learned_index_path)

    def search(self, point, radius):
        side_length_dis = self.side_length * math.sqrt(3) / 2
        small_side_length = self.side_length / math.sqrt(7)
        small_side_length_dis = self.side_length / math.sqrt(7) * math.sqrt(3) / 2

        start_time = time.time()
        cell_id = label_point(point, self.side_length, self.big_hexagon)
        cell_center_point = self.big_hexagon[cell_id]['center_point']
        big_degree = get_bear(cell_center_point, point)
        big_meter = round(distance_meter(cell_center_point, point), 0)
        # print('big_meter', big_meter)
        big_meter_plus_radius = big_meter + radius
        big_meter_sub_radius = big_meter - radius

        s_id = []
        key_range = []
        if cell_id in self.count_fractal_hexagon:
            small_label = math.ceil(
                (((big_degree + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)
            small_point = self.get_small_hexagon[round(cell_id + 0.1 * small_label, 1)]['center_point']
            to_small_meter = distance_meter(small_point, point)

            if big_meter <= to_small_meter:
                label = cell_id
                if big_meter <= radius:
                    for i in range(math.ceil(360 / self.each_degree) + 1):
                        key_min = self.c[0] * label + self.c[1] * i
                        key_max = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                else:
                    degree_range = round(math.degrees(math.asin(radius / big_meter)), 5)
                    degree_min = big_degree - degree_range
                    degree_max = big_degree + degree_range
                    degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                    degree_max_block = math.ceil((degree_max % 360) / self.each_degree)

                    if degree_min_block == degree_max_block:
                        key_min = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                    elif degree_max_block > degree_min_block:
                        for i in range(degree_max_block - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                            key_range += [key_min, key_max]

                    else:
                        for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                            key_range += [key_min, key_max]

                        for i in range(degree_max_block):
                            key_min = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * big_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * big_meter_plus_radius
                            key_range += [key_min, key_max]

                    if big_meter + radius > small_side_length_dis:
                        small_label1 = small_label
                        small_point1 = small_point
                        small_new_degree1 = get_bear(small_point1, point)
                        dis_small_point1 = distance_meter(small_point1, point)
                        to_small_meter1 = dis_small_point1 - radius
                        dis_small_point1_plus_radius = dis_small_point1 + radius

                        if to_small_meter1 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter1 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                            degree_min = small_new_degree1 - degree_range
                            degree_max = small_new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                key_max = self.c[0] * small_label1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + \
                                              self.c[2] * to_small_meter1

                                    key_max = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + \
                                              self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1
                                    key_max = self.c[0] * small_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label1 + self.c[1] * i + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                        small_label2_text = (small_label1 + 1 - 2 * (small_label1 - round(
                            (((big_degree + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60,
                            0))) % 6

                        small_label2 = 6 if small_label2_text == 0 else small_label2_text

                        small_point2 = self.get_small_hexagon[round(cell_id + 0.1 * small_label2, 1)]['center_point']
                        small_new_degree2 = get_bear(small_point2, point)
                        dis_small_point2 = distance_meter(small_point2, point)
                        to_small_meter2 = dis_small_point2 - radius
                        dis_small_point2_plus_radius = dis_small_point2 + radius

                        if to_small_meter2 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter2 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                            degree_min = small_new_degree2 - degree_range
                            degree_max = small_new_degree2 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                key_max = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2

                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

            else:
                label = round(cell_id + 0.1 * small_label, 1)
                # print('new_label', label)
                to_small_degree = get_bear(small_point, point)
                to_small_meter_sub_radius = to_small_meter - radius
                to_small_meter_plus_radius = to_small_meter + radius

                if to_small_meter <= radius:
                    for i in range(math.ceil(360 / self.each_degree) + 1):
                        key_min = self.c[0] * label + self.c[1] * i
                        key_max = self.c[0] * label + self.c[1] * i + self.c[2] * to_small_meter_plus_radius
                        key_range += [key_min, key_max]

                else:
                    degree_range = round(math.degrees(math.asin(radius / to_small_meter)), 5)
                    degree_min = to_small_degree - degree_range
                    degree_max = to_small_degree + degree_range
                    degree_min_block = math.ceil(
                        ((degree_min + 30 - math.degrees(
                            math.asin((1 / 2) / math.sqrt(7)))) % 360) / self.each_degree)
                    degree_max_block = math.ceil(
                        ((degree_max + 30 - math.degrees(
                            math.asin((1 / 2) / math.sqrt(7)))) % 360) / self.each_degree)

                    if degree_min_block == degree_max_block:
                        key_min = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * to_small_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * to_small_meter_plus_radius
                        key_range += [key_min, key_max]

                    elif degree_max_block > degree_min_block:
                        for i in range(degree_max_block - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_plus_radius
                            key_range += [key_min, key_max]
                    else:
                        for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                            key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter_plus_radius
                            key_range += [key_min, key_max]

                        for i in range(degree_max_block):
                            key_min = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * to_small_meter_sub_radius
                            key_max = self.c[0] * label + self.c[1] * (i + 1) + self.c[2] * to_small_meter_plus_radius
                            key_range += [key_min, key_max]

                    if big_meter + radius > side_length_dis:
                        new_center_degree1 = (big_degree // 60) * 60
                        new_center_point1 = destination(cell_center_point, new_center_degree1,
                                                        dist=math.sqrt(3) * self.side_length)
                        new_center_label1 = label_point(new_center_point1, self.side_length, self.big_hexagon)
                        new_degree1 = get_bear(new_center_point1, point)

                        if new_center_label1 in self.count_fractal_hexagon:
                            small_label_1 = math.ceil(
                                (((new_degree1 + 60 - math.degrees(
                                    math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                            small_point1 = self.get_small_hexagon[round(new_center_label1 + 0.1 * small_label_1, 1)][
                                'center_point']
                            dis_small_point1 = distance_meter(small_point1, point)
                            to_small_meter1 = dis_small_point1 - radius
                            dis_small_point1_plus_radius = dis_small_point1 + radius
                            if to_small_meter1 < 0:
                                for i in range(math.ceil(360 / self.each_degree) + 1):
                                    key_min = self.c[0] * label + self.c[1] * i
                                    key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            elif to_small_meter1 < small_side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                                degree_min = new_degree1 - degree_range
                                degree_max = new_degree1 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter1

                                        key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter1
                                        key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * to_small_meter1
                                        key_max = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                        key_range += [key_min, key_max]

                        else:
                            dis_new_center_point1 = distance_meter(point, new_center_point1)
                            new_label1_meter = dis_new_center_point1 - radius
                            dis_new_center_point1_plus_radius = dis_new_center_point1 + radius

                            if new_label1_meter < self.side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_new_center_point1)), 5)
                                degree_min = new_degree1 - degree_range
                                degree_max = new_degree1 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[
                                        2] * new_label1_meter
                                    key_max = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label1_meter

                                        key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label1_meter
                                        key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * new_label1_meter
                                        key_max = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * dis_new_center_point1_plus_radius
                                        key_range += [key_min, key_max]

                        new_center_degree2 = (big_degree // 60 + 1) * 60
                        new_center_point2 = destination(cell_center_point, new_center_degree2,
                                                        dist=math.sqrt(3) * self.side_length)
                        new_center_label2 = label_point(new_center_point2, self.side_length, self.big_hexagon)
                        new_degree2 = get_bear(new_center_point2, point)

                        if new_center_label2 in self.count_fractal_hexagon:
                            small_label_2 = math.ceil(
                                (((new_degree2 + 60 - math.degrees(
                                    math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                            small_point2 = self.get_small_hexagon[round(new_center_label2 + 0.1 * small_label_2, 1)][
                                'center_point']
                            dis_small_point2 = distance_meter(small_point2, point)
                            to_small_meter2 = dis_small_point2 - radius
                            dis_small_point2_plus_radius = dis_small_point2 + radius
                            if to_small_meter2 < 0:
                                for i in range(math.ceil(360 / self.each_degree) + 1):
                                    key_min = self.c[0] * label + self.c[1] * i
                                    key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            elif to_small_meter2 < small_side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                                degree_min = new_degree1 - degree_range
                                degree_max = new_degree1 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter2

                                        key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * to_small_meter2
                                        key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * to_small_meter2
                                        key_max = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                        key_range += [key_min, key_max]
                        else:
                            dis_new_center_point2 = distance_meter(point, new_center_point2)
                            new_label2_meter = dis_new_center_point2 - radius
                            dis_new_center_point2_plus_radius = dis_new_center_point2 + radius
                            if new_label2_meter < self.side_length:
                                degree_range = round(math.degrees(math.asin(radius / dis_new_center_point2)), 5)
                                degree_min = new_degree2 - degree_range
                                degree_max = new_degree2 + degree_range
                                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                                if degree_min_block == degree_max_block:
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[
                                        2] * new_label2_meter
                                    key_max = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]

                                elif degree_max_block > degree_min_block:
                                    for i in range(degree_max_block - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label2_meter

                                        key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                        key_range += [key_min, key_max]

                                else:
                                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                        key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                            2] * new_label2_meter
                                        key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                        key_range += [key_min, key_max]

                                    for i in range(degree_max_block + 1):
                                        key_min = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * new_label2_meter
                                        key_max = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * dis_new_center_point2_plus_radius
                                        key_range += [key_min, key_max]

                    elif to_small_meter + radius > small_side_length_dis:
                        small_label_1 = label

                        small_point1 = cell_center_point

                        new_degree1 = get_bear(small_point1, point)

                        dis_small_point1 = distance_meter(small_point1, point)
                        to_small_meter1 = dis_small_point1 - radius
                        dis_small_point1_plus_radius = dis_small_point1 + radius

                        if to_small_meter1 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter1 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                key_max = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1

                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                        small_label2_text = (small_label + 1 - 2 * (small_label - round(
                            (((big_degree + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60,
                            0))) % 6
                        small_label2 = 6 if small_label2_text == 0 else small_label2_text
                        small_point2 = self.get_small_hexagon[round(cell_id + 0.1 * small_label2, 1)]['center_point']
                        small_new_degree2 = get_bear(small_point2, point)
                        dis_small_point2 = distance_meter(small_point2, point)
                        to_small_meter2 = dis_small_point2 - radius
                        dis_small_point2_plus_radius = dis_small_point2 + radius
                        if to_small_meter2 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter2 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                            degree_min = small_new_degree2 - degree_range
                            degree_max = small_new_degree2 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                key_max = self.c[0] * small_label2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2

                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

        else:
            label = cell_id
            if big_meter <= radius:
                for i in range(math.ceil(360 / self.each_degree) + 1):
                    key_min = self.c[0] * label + self.c[1] * i
                    key_max = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_plus_radius
                    key_range += [key_min, key_max]

            else:
                degree_range = round(math.degrees(math.asin(radius / big_meter)), 5)
                degree_min = big_degree - degree_range
                degree_max = big_degree + degree_range
                degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                if degree_min_block == degree_max_block:
                    key_min = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_sub_radius
                    key_max = self.c[0] * label + self.c[1] * degree_min_block + self.c[2] * big_meter_plus_radius
                    key_range += [key_min, key_max]

                elif degree_max_block > degree_min_block:
                    for i in range(degree_max_block - degree_min_block + 1):
                        key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                else:
                    for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                        key_min = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * (degree_min_block + i) + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                    for i in range(degree_max_block + 1):
                        key_min = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_sub_radius
                        key_max = self.c[0] * label + self.c[1] * i + self.c[2] * big_meter_plus_radius
                        key_range += [key_min, key_max]

                if big_meter + radius > side_length_dis:
                    new_center_degree1 = (big_degree // 60) * 60
                    new_center_point1 = destination(cell_center_point, new_center_degree1,
                                                    dist=math.sqrt(3) * self.side_length)
                    new_center_label1 = label_point(new_center_point1, self.side_length, self.big_hexagon)
                    new_degree1 = get_bear(new_center_point1, point)

                    if new_center_label1 in self.count_fractal_hexagon:
                        small_label_1 = math.ceil(
                            (((new_degree1 + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                        small_point1 = self.get_small_hexagon[round(new_center_label1 + 0.1 * small_label_1, 1)][
                            'center_point']
                        dis_small_point1 = distance_meter(small_point1, point)
                        to_small_meter1 = dis_small_point1 - radius
                        dis_small_point1_plus_radius = dis_small_point1 + radius

                        if to_small_meter1 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter1 < small_side_length:
                            # print('dis_small_point1', radius, dis_small_point1)
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point1)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * to_small_meter1
                                key_max = self.c[0] * small_label_1 + self.c[1] * degree_min_block + self.c[2] * dis_small_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * to_small_meter1

                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * to_small_meter1
                                    key_max = self.c[0] * small_label_1 + self.c[1] * i + self.c[2] * dis_small_point1_plus_radius
                                    key_range += [key_min, key_max]

                    else:
                        dis_new_center_point1 = distance_meter(point, new_center_point1)
                        new_label1_meter = dis_new_center_point1 - radius
                        dis_new_center_point1_plus_radius = dis_new_center_point1 + radius

                        if new_label1_meter < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label +self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_new_center_point1_plus_radius
                                key_range += [key_min, key_max]

                        elif new_label1_meter < self.side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_new_center_point1)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[2] * new_label1_meter
                                key_max = self.c[0] * new_center_label1 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point1_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label1_meter

                                    key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label1_meter
                                    key_max = self.c[0] * new_center_label1 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * new_label1_meter
                                    key_max = self.c[0] * new_center_label1 + self.c[1] * i + self.c[2] * dis_new_center_point1_plus_radius
                                    key_range += [key_min, key_max]

                    new_center_degree2 = (big_degree // 60 + 1) * 60
                    new_center_point2 = destination(cell_center_point, new_center_degree2, math.sqrt(3) * self.side_length)
                    new_center_label2 = label_point(new_center_point2, self.side_length, self.big_hexagon)
                    new_degree2 = get_bear(new_center_point2, point)

                    if new_center_label2 in self.count_fractal_hexagon:
                        small_label_2 = math.ceil(
                            (((new_degree2 + 60 - math.degrees(math.asin((1 / 2) / math.sqrt(7)))) + 360) % 360) / 60)

                        small_point2 = self.get_small_hexagon[round(new_center_label2 + 0.1 * small_label_2, 1)][
                            'center_point']
                        dis_small_point2 = distance_meter(small_point2, point)
                        to_small_meter2 = dis_small_point2 - radius
                        dis_small_point2_plus_radius = dis_small_point2 + radius

                        if to_small_meter2 < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif to_small_meter2 < small_side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_small_point2)), 5)
                            degree_min = new_degree1 - degree_range
                            degree_max = new_degree1 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * to_small_meter2
                                key_max = self.c[0] * small_label_2 + self.c[1] * degree_min_block + self.c[2] * dis_small_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2

                                    key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * to_small_meter2
                                    key_max = self.c[0] * small_label_2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]

                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * to_small_meter2
                                    key_max = self.c[0] * small_label_2 + self.c[1] * i + self.c[2] * dis_small_point2_plus_radius
                                    key_range += [key_min, key_max]
                    else:
                        dis_new_center_point2 = distance_meter(new_center_point2, point)
                        new_label2_meter = dis_new_center_point2 - radius
                        dis_new_center_point2_plus_radius = dis_new_center_point2 + radius

                        if new_label2_meter < 0:
                            for i in range(math.ceil(360 / self.each_degree) + 1):
                                key_min = self.c[0] * label + self.c[1] * i
                                key_max = self.c[0] * label + self.c[1] * i + self.c[2] * dis_new_center_point2_plus_radius
                                key_range += [key_min, key_max]

                        elif new_label2_meter < self.side_length:
                            degree_range = round(math.degrees(math.asin(radius / dis_new_center_point2)), 5)
                            degree_min = new_degree2 - degree_range
                            degree_max = new_degree2 + degree_range
                            degree_min_block = math.ceil((degree_min % 360) / self.each_degree)
                            degree_max_block = math.ceil((degree_max % 360) / self.each_degree)
                            if degree_min_block == degree_max_block:
                                key_min = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[2] * new_label2_meter
                                key_max = self.c[0] * new_center_label2 + self.c[1] * degree_min_block + self.c[2] * dis_new_center_point2_plus_radius
                                key_range += [key_min, key_max]

                            elif degree_max_block > degree_min_block:
                                for i in range(degree_max_block - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label2_meter

                                    key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]

                            else:
                                for i in range(int(360 / self.each_degree) - degree_min_block + 1):
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[
                                        2] * new_label2_meter
                                    key_max = self.c[0] * new_center_label2 + self.c[1] * (degree_min_block + i) + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]
                                for i in range(degree_max_block + 1):
                                    key_min = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * new_label2_meter
                                    key_max = self.c[0] * new_center_label2 + self.c[1] * i + self.c[2] * dis_new_center_point2_plus_radius
                                    key_range += [key_min, key_max]

        pos = self.key_model._find_insert_position(key_range)

        for i in range(int(len(pos)/2)):
            pos_min = pos[i * 2]
            pos_max = pos[i * 2 + 1]
            if pos_min == pos_max and pos_min % 2 != 0:
                pass
            else:
                idx1 = pos_min if pos_min % 2 == 0 else pos_min + 1
                idx2 = pos_max if pos_max % 2 == 0 else pos_max - 1

                s_id += [self.key_dict[int(idx1) + j * 2]['link_id'] for j in range(int((idx2 - idx1) / 2) + 1)]

        new_s_id = [link.split('_')[0] for link in s_id]
        new_s_id = list(np.unique(new_s_id))
        candidate_segment = []

        for link in new_s_id:
            link_start = self.original_segment[link]['node_a_id']
            link_end = self.original_segment[link]['node_b_id']
            d, p, _ = distance_point_to_segment(point, self.original_segment[link]['node_a'],
                                                self.original_segment[link]['node_b'])
            if d <= radius:
                candidate_segment.append([link, link_start, link_end, p, d])
        candidate_segment.sort(key=lambda x: x[-1])
        candidate_time = round(time.time() - start_time, 9)
        # print(len(candidate_segment))
        return candidate_segment, candidate_time


class UseLearnedIndex_200(object):
    def __init__(self, config):
        self.config = config
        self.all_segment_path = np.load(self.config.data_dir + '/segment_all_dict.npy', allow_pickle=True)
        self.all_segment = self.all_segment_path.item()
        self.original_segment_path = np.load(self.config.data_dir + '/original_segment.npy', allow_pickle=True)
        self.original_segment = self.original_segment_path.item()
        self.learned_index_path = self.config.data_dir + '/learned_index'
        self.all_key = np.load(self.config.data_dir + '/hexagon/all_key.npy', allow_pickle=True)
        self.side_length = 200
        self.big_hexagon_path = np.load(self.config.data_dir + '/hexagon/big_hexagon.npy', allow_pickle=True)
        self.big_hexagon = self.big_hexagon_path.item()
        self.c = [10 ** (3 + len(str(self.side_length))), 10 ** len(str(self.side_length)), 1]

        self.key_dict_path = np.load(self.config.data_dir + '/hexagon/key_dict.npy', allow_pickle=True)
        self.key_dict = self.key_dict_path.item()

        self.construct()
        self.key_model = load_db(self.learned_index_path + '/temp_db.p')

    def construct(self):
        if not os.path.exists(self.learned_index_path):
            os.makedirs(self.learned_index_path)
            key_dataset = own_dataset(self.all_key)
            train(key_dataset, self.learned_index_path)

    def search(self, point, radius):
        side_length_dis = self.side_length * math.sqrt(3) / 2
        start_time = time.time()
        cell_id = label_point(point, self.side_length, self.big_hexagon)
        cell_center_point = self.big_hexagon[cell_id]['center_point']
        big_degree = get_bear(cell_center_point, point)
        big_meter = round(distance_meter(cell_center_point, point), 0)
        # print('big_meter', big_meter)

        s_id = []
        key_range = []

        label = cell_id
        if big_meter + radius <= side_length_dis:
            key_min = self.c[0] * label
            key_max = self.c[0] * (label + 1)
            key_range += [key_min, key_max]

        else:
            block1 = big_degree // 60
            degree_block1 = block1 if block1 != 0 else 6

            # raw hexagon
            key_min = self.c[0] * label + self.c[1] * degree_block1
            key_max = self.c[0] * label + self.c[1] * (degree_block1 + 2)
            key_range += [key_min, key_max]

            if degree_block1 == 6:
                key_min = self.c[0] * label
                key_max = self.c[0] * label + self.c[1] * 2

                key_range += [key_min, key_max]

            #
            new_center_degree1 = block1 * 60
            new_center_point1 = destination(cell_center_point, new_center_degree1, dist=math.sqrt(3) * self.side_length)
            new_center_label1 = label_point(new_center_point1, self.side_length, self.big_hexagon)

            new_degree_block = (degree_block1 + 3) % 6
            new_degree_block1 = new_degree_block if new_degree_block != 0 else 6

            # the neighbor hexagon 1
            key_min = self.c[0] * new_center_label1 + self.c[1] * (new_degree_block1 - 1)
            key_max = self.c[0] * new_center_label1 + self.c[1] * (new_degree_block1 + 1)

            key_range += [key_min, key_max]

            if degree_block1 == 4:
                key_min = self.c[0] * new_center_label1 + self.c[1] * 6
                key_max = self.c[0] * (new_center_label1 + 1)

                key_range += [key_min, key_max]

            #
            block2 = block1 + 1
            degree_block2 = degree_block1 + 1 if block1 != 0 else 1

            new_center_degree2 = block2 * 60
            new_center_point2 = destination(cell_center_point, new_center_degree2, math.sqrt(3) * self.side_length)
            new_center_label2 = label_point(new_center_point2, self.side_length, self.big_hexagon)

            new_degree_block2 = (degree_block2 + 3) % 6
            new_degree_block22 = new_degree_block2 if new_degree_block2 != 0 else 6

            # the neighbor hexagon 2
            key_min = self.c[0] * new_center_label2 + self.c[1] * new_degree_block22
            key_max = self.c[0] * new_center_label2 + self.c[1] * (new_degree_block22 + 2)

            key_range += [key_min, key_max]

            if degree_block1 == 2:
                key_min = self.c[0] * new_center_label2
                key_max = self.c[0] * new_center_label2 + self.c[1] * 2
                key_range += [key_min, key_max]

        pos = self.key_model._find_insert_position(key_range)

        for i in range(int(len(pos)/2)):
            pos_min = pos[i * 2]
            pos_max = pos[i * 2 + 1]
            if pos_min == pos_max and pos_min % 2 != 0:
                pass
            else:
                idx1 = pos_min if pos_min % 2 == 0 else pos_min + 1
                idx2 = pos_max if pos_max % 2 == 0 else pos_max - 1

                s_id += [self.key_dict[int(idx1) + j * 2]['link_id'] for j in range(int((idx2 - idx1) / 2) + 1)]

        new_s_id = [link.split('_')[0] for link in s_id]
        new_s_id = list(np.unique(new_s_id))
        candidate_segment = []

        for link in new_s_id:
            link_start = self.original_segment[link]['node_a_id']
            link_end = self.original_segment[link]['node_b_id']
            d, p, _ = distance_point_to_segment(point, self.original_segment[link]['node_a'],
                                                self.original_segment[link]['node_b'])
            if d <= radius:
                candidate_segment.append([link, link_start, link_end, p, d])
        candidate_segment.sort(key=lambda x: x[-1])
        candidate_time = round(time.time() - start_time, 9)
        # print(len(candidate_segment))
        return candidate_segment, candidate_time
