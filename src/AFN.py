# coding=utf-8
import shutil
import os
import glob
from datetime import date, timedelta
import random

import numpy as np
import tensorflow as tf

from utils import get_third_nearest_checkpoint

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("feature_size", 5500, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 10, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
tf.app.flags.DEFINE_integer("hidden_size", 600, "hidden unit size")
tf.app.flags.DEFINE_integer("instance_size", 288609, "Number of instances in the dataset")
tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '400,400,400', "deep layers")
tf.app.flags.DEFINE_string("dropout", '1.,1.,1.', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '../data/frappe/', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", './checkpoint/',"model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
tf.app.flags.DEFINE_boolean("ensemble", True, "whether to use the ensemble version")

def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)

    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels
                
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(
        5000000)  # multi-thread pre-process then prefetch

    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    learning_rate = params["learning_rate"]
    layers  = list(map(int, params["deep_layers"].split(',')))
    layers_dnn = [400,400,400]
    dropout = list(map(float, params["dropout"].split(',')))

    # ------bulid weights------
    Feat_Emb = tf.get_variable(name="h_lr_emb", shape=[feature_size, embedding_size],
                  initializer=tf.glorot_normal_initializer())
    Feat_Emb_deep = tf.get_variable(name="h_lr_emb_deep", shape=[feature_size, embedding_size],
                               initializer=tf.random_normal_initializer(stddev=0.1), )
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    # FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(1.0))
    Feat_Emb = tf.abs(Feat_Emb)
    Feat_Emb = tf.clip_by_value(Feat_Emb, 1e-4, np.infty)
    # ------build feaure-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])
    feat_vals = tf.clip_by_value(feat_vals, 0.001, 1.)

    # ------build f(x)------
    with tf.variable_scope("Permutation-Layer"):
        embeddings_origin = tf.nn.embedding_lookup(Feat_Emb, feat_ids)  # None * F * K
        embeddings_origin_deep = tf.nn.embedding_lookup(Feat_Emb_deep, feat_ids)  # None * F * K
        tf.summary.histogram('embeddings_origin', embeddings_origin)
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F *1
        embeddings = tf.multiply(embeddings_origin, feat_vals)
        embeddings_deep = tf.multiply(embeddings_origin_deep, feat_vals)
        print("shape of embeddings: %s" % embeddings.get_shape().as_list())
        embeddings_trans = tf.transpose(embeddings, perm=[0, 2, 1])  # None * K * F
        embeddings_trans = tf.log(embeddings_trans, name="log_input")
        embeddings_trans = tf.check_numerics(embeddings_trans, "log2")
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False
        embeddings_trans = batch_norm_layer(embeddings_trans, train_phase=train_phase,
                                       scope_bn='bn_log')
    with tf.variable_scope("Layer-1"):
        hidden_size = FLAGS.hidden_size
        weights = tf.get_variable("h_lr_weights", shape=[field_size, hidden_size],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [hidden_size], initializer=tf.constant_initializer(0))
        layer1 = tf.einsum('bkf,fo->bko', embeddings_trans, weights) + biases

    with tf.variable_scope("Prediction"):
        interactions = tf.exp(layer1, name="restored_input")  # None*K*O
        interactions = batch_norm_layer(interactions, train_phase=train_phase,
                                       scope_bn='bn_inter')
        print(interactions.get_shape().as_list())
        interactions = tf.reshape(interactions, shape=[-1, FLAGS.embedding_size*FLAGS.hidden_size])  # None * (K*O)
        with tf.variable_scope("Deep-part"):
            deep_inputs = interactions
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[0])  # None * K
            for i in range(len(layers)):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                                                                scope='mlp%d' % i)

                if FLAGS.batch_norm:
                    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase,
                                                   scope_bn='bn_%d' % i)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[
                        i])

            y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                                       scope='deep_out')
            y_afn = tf.squeeze(tf.reshape(y_deep, shape=[-1]))

    if FLAGS.ensemble:
        with tf.variable_scope("Deep-part"):
            deep_inputs = tf.reshape(embeddings_deep, shape=[-1, field_size * embedding_size])  # None * (F*K)
            for i in range(len(layers_dnn)):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers_dnn[i], \
                                                                scope='mlp%d' % i)
                if FLAGS.batch_norm:
                    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase,
                                                   scope_bn='bn_%d' % i)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[
                        i])  # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)

            y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                                       scope='deep_out')
            y_d = tf.reshape(y_deep, shape=[-1])

        w1 = tf.get_variable(name='w1', shape=[1], initializer=tf.constant_initializer(0.5))
        w2 = tf.get_variable(name='w2', shape=[1], initializer=tf.constant_initializer(0.5))
        b_p =  tf.get_variable(name='b_p', shape=[1], initializer=tf.constant_initializer(0.0))
        y_1 = w1*tf.stop_gradient(y_d)
        y_2 = w2*tf.stop_gradient(y_afn)
        y =  y_1 + y_2 + b_p
    else:
        y = y_afn
    pred = tf.sigmoid(y)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------mk
    if FLAGS.ensemble:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))\
        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d, labels=labels))\
        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_afn, labels=labels))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

    # Provide an estimator spec for `ModeKeys.EVAL`
    log_loss = tf.losses.log_loss(labels, pred)
    auc_metric = tf.metrics.auc(labels, pred)
    loss_metric = tf.metrics.mean(log_loss)
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred),
        "logloss": tf.metrics.mean(log_loss),
        "stop_criterion": (auc_metric[0]-loss_metric[0], tf.group(auc_metric[1],loss_metric[1]))
    }

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    gvs = optimizer.compute_gradients(loss)

    def ClipIfNotNone(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(clipped_gradients, global_step=tf.train.get_global_step())

    for grad, var in gvs:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op)

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
                          
    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('ensemble ', FLAGS.ensemble)

    # ------init Envs------
    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % FLAGS.data_dir)
    print("te_files:", te_files)

    with open(tr_files[0]) as f:
        train_size = sum(1 for line in f)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    # set_dist_env()

    # ------bulid Tasks------
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        # "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(gpu_options=gpu_options),
                                              log_step_count_steps=100, save_summary_steps=100,
                                              save_checkpoints_steps=train_size // FLAGS.batch_size + 1)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, "stop_criterion", max_steps_without_increase=train_size // FLAGS.batch_size,
                                                         run_every_secs=None, run_every_steps=10)
    os.makedirs(estimator.eval_dir())
    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=None, batch_size=FLAGS.batch_size, perform_shuffle=True),
            hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None,
            start_delay_secs=10, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print("Early stopped, start evaluation...")
        estimator.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                           checkpoint_path=get_third_nearest_checkpoint(estimator.model_dir))
    elif FLAGS.task_type == 'eval':
        estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                               predict_keys="prob")
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
