import numpy as np
import pandas as pd
import tensorflow as tf

#If you do not have the files in the same directory as the model, please change.
df_train = 'datatrain.csv'
df_label = 'labelstrain.csv'
df_eval_x = 'dataeval.csv'
df_eval_y = 'labelseval.csv'



#This is the data pipeline.
X_train = pd.read_csv(df_train, header=None).values # Training input data as Numpy array
y_train = pd.read_csv(df_label, header=None).values.ravel()
X_eval = pd.read_csv(df_eval_x, header=None).values
y_eval = pd.read_csv(df_eval_y, header=None).values.ravel()
y_train = np.asarray(y_train, dtype=np.int32)
X_train = np.asarray(X_train, dtype=np.float)
X_eval = np.asarray(X_eval, dtype=np.float)

#This is the whiting part, please comment out if you do not want this effect.
mean = np.mean(X_train)
std = np.std(X_train)

X_train -= np.mean(X_train)
X_train /= np.std(X_train)

X_eval -= np.mean(X_eval)
X_eval /= np.std(X_eval)


print(X_train.shape, y_train.shape, X_eval.shape, y_eval.shape)
print(X_train.dtype, y_train.dtype)
np.unique(y_eval)


np.max(y_train)


#Model defined
def cnn_model_fn(features, labels, mode):

  # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 25, 25, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    pool2_flat = tf.reshape(pool2, [-1, 6 * 6 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
         loss=loss,
         global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
       "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
       "auc": tf.metrics.auc(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          num_thresholds=200,
          metrics_collections=None,
          updates_collections=None,
          curve='ROC',
          name=None,
          summation_method='trapezoidal'
          ),
       "false_negatives": tf.metrics.false_negatives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
),
        "false_positives": tf.metrics.false_positives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
),
        "true_negatives":tf.metrics.true_negatives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
),
        "true_positives":tf.metrics.true_positives(
          labels=labels,
          predictions=predictions["classes"],
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None
)


}
    return tf.estimator.EstimatorSpec(
       mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)


# Create the Estimator
test_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,model_dir="enter where you would like to save the model...")


#change the number if you want to have it run longer or shorter.
total = 1000

test_results=[total]
train_resultsarray=[total]
# Train the model
for n in range(total):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=10,
    shuffle=True)
    test_classifier.train(
    input_fn=train_input_fn,
    steps=2000,
    hooks=[logging_hook])

    print("n is right now " + str(n))


# Evaluate the model and print results, if statement here
    if n % 100 == 0:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_eval},
    y=y_eval,
    num_epochs=1,
    shuffle=True)
        eval_results = test_classifier.evaluate(input_fn=eval_input_fn)
        print("Testing/Validation data results:")
        print(eval_results)
        test_results.append(eval_results)

        traineval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)
        traineval_results = test_classifier.evaluate(input_fn=traineval_input_fn)
        print("Training data results:")
        print(traineval_results)
        train_resultsarray.append(traineval_results)

print(type(test_results))
print(type(train_resultsarray))

import pprint
print("Test/Validation results:")
pprint.pprint(test_results)
print("Train results:")
pprint.pprint(train_resultsarray)

saver = tf.train.Saver()
save_path = saver.save(session, "enter where you would like to save the model...") #Specify where to save the model
print("Saved model at: ", save_path)

with open('results_test.csv', 'w') as f:
    for item in train_results:
        f.write("%s\n" % item)

with open('results_train.csv', 'w') as f:
    for item in train_resultsarray:
        f.write("%s\n" % item)
