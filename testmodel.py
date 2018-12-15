import tensorflow as tf
import numpy as np
import pandas as pd
import pprint

df_eval_x = 'datatest.csv'
df_eval_y = 'labelstest.csv'


df_x_cord = 'xmatlabcord.csv'
df_y_cord = 'ymatlabcord.csv'

X_test = pd.read_csv(df_eval_x, header=None).values
y_test = pd.read_csv(df_eval_y, header=None).values.ravel()
X_test = np.asarray(X_test, dtype=np.float)
y_test = np.asarray(y_test, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float)

print(X_test.dtype, y_test.dtype)
np.unique(y_test)

y_test[y_test == 1] = 0
y_test[y_test == 2] = 1

np.max(y_test)

print(X_test.shape, y_test.shape)

xcords = pd.read_csv(df_x_cord, header=None).values.ravel()
ycords = pd.read_csv(df_y_cord, header=None).values.ravel()
xcords = np.asarray(xcords, dtype=np.int32)
ycords = np.asarray(ycords, dtype=np.int32)

def cnn_model_fn(features, labels, mode):

  # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 25, 25, 1])
  # Convolutional Layer #1

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

  # Pooling Layer #1

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
  # Pooling Layer #2

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer

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
        predict_op = ()
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
        "false_postives": tf.metrics.false_positives(
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
        "true_postivies":tf.metrics.true_positives(
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
    model_fn=cnn_model_fn,model_dir="enter model location here...")

test_results=[len(y_test)]#end*steps]

x_example = X_test.shape
print(type(x_example))
totalentries = x_example[0]
print(totalentries)
print(type(totalentries))

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

def evaluate(X_test,y_test):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
    eval_results = test_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)
    test_results.append(eval_results)

predict_results=[len(y_test)]#end*steps]

x_example = X_test.shape
print(type(x_example))
totalentriesp = x_example[0]
print(totalentriesp)
print(type(totalentriesp))

def predict(X_test):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    num_epochs=1,
    shuffle=False)
    return list(test_classifier.predict(input_fn=predict_input_fn,yield_single_examples=True))


indexnumber = []
p0list = []
p1list = []

def gettop30(predict_results):
    for item in predict_results:
        c = item["classes"]
        p = item["probabilities"]
        p0 = p.item(0)
        p1 = p.item(1)
        p0list.append(p0)
        p1list.append(p1)

    print("p0")
    print( sorted( [(x,i) for (i,x) in enumerate(p0list)], reverse=True )[:30] )
    print("p1")
    print( sorted( [(x,i) for (i,x) in enumerate(p1list)], reverse=True )[:30] )

def getonly1(predict_results):
    i = 0
    onlyone = []
    indexnumber = []
    for item in predict_results:
        c = item["classes"]
        if c == 1:
            indexnumber.append(i)
            onlyone.append(item)
            itemindex = i
            print(itemindex)
            indexnumber.append(i)
        i += 1

    return onlyone,indexnumber


def firstfiler(predict_results):
    predict_results_filter=[]
    indexnumber = []
    s = predict_results[1]
    print(type(s["classes"]))
    i = 0
    for item in predict_results:
        c = item["classes"]
        p = item["probabilities"]
        p0 = p.item(0)
        p1 = p.item(1)

        v = .999999999999

        if c == 1:
            if p0 < 1-v and p1 > v:
                predict_results_filter.append(item)
                itemindex = i
                print(itemindex)
                indexnumber.append(i)
        i += 1

    for item in predict_results_filter:
        print("the filter got")
        print(item)

    return predict_results_filter,indexnumber

    with open('firstfilter_results.csv', 'w') as f:
        for item in predict_results_filter:#l:
            f.write("%s\n" % item)

selectedxy = []
def getxandy(indexnumber):
    for item in indexnumber:
        x = xcords[item]
        y = ycords[item]
        selectedxy.append(str(x)+ "," + str(y))

    return selectedxy

with tf.Session() as sess:
    loader = tf.train.import_meta_graph('enter here location of meta file...')
    loader.restore(sess, tf.train.latest_checkpoint('enter here location of meta file...'))

    predict_results = predict(X_test)

onlyone,indexnumber=getonly1(predict_results)
predict_results_filter1,indexnumber=list(firstfiler(predict_results))
print("Filter results:")
pprint.pprint(predict_results_filter1)

with open('onlyone.txt', 'w') as f:
    for item in onlyone:
        f.write("%s\n" % item)

with open('firstfilter.txt', 'w') as f:
    for item in predict_results_filter1:
        f.write("%s\n" % item)

selectedxy = getxandy(indexnumber)
print(selectedxy)
with open('onlyone_cords.txt', 'w') as f:
    for item in selectedxy:#l:
        f.write("%s\n" % item)
