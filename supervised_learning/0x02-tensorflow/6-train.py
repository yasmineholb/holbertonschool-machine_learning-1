#!/usr/bin/env python3
""" train function """
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ train function """
    classes = Y_train.shape[1]
    nx = X_train.shape[1]
    X, y = create_placeholders(nx, classes)
    y_pred = forward_prop(X, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('X', X)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    init = tf.global_variables_initializer()
    init1 = tf.local_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    init.run()
    init1.run()
    for i in range(iterations+1):
        loss_train, accuracy_train = sess.run((loss, accuracy), feed_dict={
            X: X_train,
            y: Y_train
        })
        loss_valid, accuracy_valid = sess.run((loss, accuracy), feed_dict={
            X: X_valid,
            y: Y_valid
        })
        sess.run(train_op, feed_dict={
            X: X_train,
            y: Y_train
        })
        if(i == 0) or (i % 100 == 0) or (i == iterations):
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(loss_valid))
            print("\tValidation Accuracy: {}".format(
                accuracy_valid))
    saver.save(sess, save_path)
    return save_path
