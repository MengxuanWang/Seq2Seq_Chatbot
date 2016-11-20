import time
import os
import sys
import math

import numpy as np
import tensorflow as tf

import utils
from model import ChatbotModel

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.5, "Learning rate")
flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
flags.DEFINE_integer("max_epoch", 6, "Maximum number of times to go over training set")
flags.DEFINE_integer("hidden_size", 128, "Size of each model layer")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model")
flags.DEFINE_integer("vocab_size", 40000, "Max vocabulary size")
flags.DEFINE_integer("dropout", 0.5, "Probability of hidden inputs being removed between 0 and 1")
flags.DEFINE_integer("num_samples", 1024, "Number of samples for sampled softmax loss")
flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit)")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint")
flags.DEFINE_string("data_dir", "./data", "Directory containing processed data")
flags.DEFINE_string("train_dir", "./tmp", "Training directory")
flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32")
flags.DEFINE_boolean("use_lstm", True, "Using LSTM or GRU")
flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding")

FLAGS = flags.FLAGS

#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(5, 15), (10, 20), (20, 25), (30, 40), (40, 50)]
textloader = utils.TextLoader(FLAGS.data_dir)

def create_model(session, forward_only):
    dropout =  FLAGS.dropout if not FLAGS.decode else 1.0
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = ChatbotModel(
        vocab_size=textloader.vocab_size,
        buckets=_buckets,
        hidden_size=FLAGS.hidden_size,
        dropout=dropout,
        num_layers=FLAGS.num_layers,
        max_gradient_norm=FLAGS.max_gradient_norm,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print "Reading model parameters from %s ... " % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Creating model with fresh parameters ... "
        session.run(tf.initialize_all_variables())
    return model

def read_data(source_tensors, target_tensors, max_size=None):
    data_set = [[] for _ in _buckets]
    counter = 0
    total_num = source_tensors.shape[0]
    while counter < total_num and (not max_size or counter < max_size):
        source, target = source_tensors[counter], target_tensors[counter]
        if counter % 10000 == 0:
            print "reading data line %d" % counter
            sys.stdout.flush()
        target.append(utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source) < source_size and len(target) < target_size:
                data_set[bucket_id].append([source, target])
                break
        counter += 1
    return data_set

def train():
    with tf.Session() as sess:
        # Create model.
        print "Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size)
        model = create_model(sess, False)

        # Read data into buckets and compute their size.
        train_set = read_data(textloader.tensor_sources,
                              textloader.tensor_targets,
                              FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print "global step %d learning rate %.4f step-time %.2f perplexity %.2f"\
                     % (model.global_step.eval(), model.learning_rate.eval(),
                              step_time, perplexity)
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss
                checkpoint_path = os.path.join(FLAGS.train_dir, "chatbot.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

def decode():
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            token_ids = sentence_to_token_ids(sentence)
            #print token_ids
            bucket_id = min([b for b in xrange(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS sysbol in outputs, cut them at that point.
            if utils.EOS_ID in outputs:
                outputs = outputs[: outputs.index(utils.EOS_ID)]
            print "".join([textloader.chars[output] for output in outputs])
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def sentence_to_token_ids(sentence):
    sentence = unicode(sentence, 'utf-8')
    chars = list(sentence.strip())
    return [textloader.vocab.get(char, utils.UNK_ID) for char in chars]

def main():
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    main()
