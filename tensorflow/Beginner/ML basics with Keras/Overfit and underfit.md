### Overfit and underfit

As always, the code in this example will use the [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) API, which you can learn more about in the TensorFlow [Keras guide](https://www.tensorflow.org/guide/keras).

In both of the previous examples—[classifying text](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) and [predicting fuel efficiency](https://www.tensorflow.org/tutorials/keras/regression)—the accuracy of models on the validation data would peak after training for a number of epochs and then stagnate or start decreasing.

In other words, your model would *overfit* to the training data. Learning how to deal with overfitting is important. Although it's often possible to achieve high accuracy on the *training set*, what you really want is to develop models that generalize well to a *testing set* (or data they haven't seen before).

The opposite of overfitting is *underfitting*. Underfitting occurs when there is still room for improvement on the train data. This can happen for a number of reasons: If the model is not powerful enough, is over-regularized, or has simply not been trained long enough. This means the network has not learned the relevant patterns in the training data.

If you train for too long though, the model will start to overfit and learn patterns from the training data that don't generalize to the test data. You need to strike a balance. Understanding how to train for an appropriate number of epochs as you'll explore below is a useful skill.

To prevent overfitting, the best solution is to use more complete training data. The dataset should cover the full range of inputs that the model is expected to handle. Additional data may only be useful if it covers new and interesting cases.

A model trained on more complete data will naturally generalize better. When that is no longer possible, the next best solution is to use techniques like regularization. These place constraints on the quantity and type of information your model can store. If a network can only afford to memorize a small number of patterns, the optimization process will force it to focus on the most prominent patterns, which have a better chance of generalizing well.

In this notebook, you'll explore several common regularization techniques, and use them to improve on a classification model.

## Setup

Before getting started, import the necessary packages:

```python
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)
2.8.0
!pip install git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
```

## The Higgs dataset

The goal of this tutorial is not to do particle physics, so don't dwell on the details of the dataset. It contains 11,000,000 examples, each with 28 features, and a binary class label.

```python
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
Downloading data from http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz
2816409600/2816407858 [==============================] - 156s 0us/step
2816417792/2816407858 [==============================] - 156s 0us/step
FEATURES = 28
```

The [`tf.data.experimental.CsvDataset`](https://www.tensorflow.org/api_docs/python/tf/data/experimental/CsvDataset) class can be used to read csv records directly from a gzip file with no intermediate decompression step.

```python
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")
```

That csv reader class returns a list of scalars for each record. The following function repacks that list of scalars into a (feature_vector, label) pair.

```python
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label
```

TensorFlow is most efficient when operating on large batches of data.

So, instead of repacking each row individually make a new [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) that takes batches of 10,000 examples, applies the `pack_row` function to each batch, and then splits the batches back up into individual records:

```python
packed_ds = ds.batch(10000).map(pack_row).unbatch()
```

Inspect some of the records from this new `packed_ds`.

The features are not perfectly normalized, but this is sufficient for this tutorial.

```python
for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)
tf.Tensor(
[ 0.8692932  -0.6350818   0.22569026  0.32747006 -0.6899932   0.75420225
 -0.24857314 -1.0920639   0.          1.3749921  -0.6536742   0.9303491
  1.1074361   1.1389043  -1.5781983  -1.0469854   0.          0.65792954
 -0.01045457 -0.04576717  3.1019614   1.35376     0.9795631   0.97807616
  0.92000484  0.72165745  0.98875093  0.87667835], shape=(28,), dtype=float32)
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_TfcXuv33Fvka_1.png)

To keep this tutorial relatively short, use just the first 1,000 samples for validation, and the next 10,000 for training:

```python
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
```

The [`Dataset.skip`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip) and [`Dataset.take`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take) methods make this easy.

At the same time, use the [`Dataset.cache`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache) method to ensure that the loader doesn't need to re-read the data from the file on each epoch:

```python
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
train_ds
<CacheDataset element_spec=(TensorSpec(shape=(28,), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.float32, name=None))>
```

These datasets return individual examples. Use the [`Dataset.batch`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) method to create batches of an appropriate size for training. Before batching, also remember to use [`Dataset.shuffle`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) and [`Dataset.repeat`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat) on the training set.

```python
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
```

## Demonstrate overfitting

The simplest way to prevent overfitting is to start with a small model: A model with a small number of learnable parameters (which is determined by the number of layers and the number of units per layer). In deep learning, the number of learnable parameters in a model is often referred to as the model's "capacity".

Intuitively, a model with more parameters will have more "memorization capacity" and therefore will be able to easily learn a perfect dictionary-like mapping between training samples and their targets, a mapping without any generalization power, but this would be useless when making predictions on previously unseen data.

Always keep this in mind: deep learning models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting.

On the other hand, if the network has limited memorization resources, it will not be able to learn the mapping as easily. To minimize its loss, it will have to learn compressed representations that have more predictive power. At the same time, if you make your model too small, it will have difficulty fitting to the training data. There is a balance between "too much capacity" and "not enough capacity".

Unfortunately, there is no magical formula to determine the right size or architecture of your model (in terms of the number of layers, or the right size for each layer). You will have to experiment using a series of different architectures.

To find an appropriate model size, it's best to start with relatively few layers and parameters, then begin increasing the size of the layers or adding new layers until you see diminishing returns on the validation loss.

Start with a simple model using only densely-connected layers ([`tf.keras.layers.Dense`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) as a baseline, then create larger models, and compare them.

### Training procedure

Many models train better if you gradually reduce the learning rate during training. Use [`tf.keras.optimizers.schedules`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules) to reduce the learning rate over time:

```python
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)
```

The code above sets a [`tf.keras.optimizers.schedules.InverseTimeDecay`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay) to hyperbolically decrease the learning rate to 1/2 of the base rate at 1,000 epochs, 1/3 at 2,000 epochs, and so on.

```python
step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_HIo_yPjEAFgn_0.png)

Each model in this tutorial will use the same training configuration. So set these up in a reusable way, starting with the list of callbacks.

The training for this tutorial runs for many short epochs. To reduce the logging noise use the `tfdocs.EpochDots` which simply prints a `.` for each epoch, and a full set of metrics every 100 epochs.

Next include [`tf.keras.callbacks.EarlyStopping`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) to avoid long and unnecessary training times. Note that this callback is set to monitor the `val_binary_crossentropy`, not the `val_loss`. This difference will be important later.

Use [`callbacks.TensorBoard`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) to generate TensorBoard logs for the training.

```python
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]
```

Similarly each model will use the same [`Model.compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) and [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) settings:

```python
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history
```

### Tiny model

Start by training a model:

```python
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 16)                464       
                                                                 
 dense_1 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 481
Trainable params: 481
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.4896,  binary_crossentropy:0.8395,  loss:0.8395,  val_accuracy:0.5070,  val_binary_crossentropy:0.7700,  val_loss:0.7700,  
....................................................................................................
Epoch: 100, accuracy:0.6016,  binary_crossentropy:0.6240,  loss:0.6240,  val_accuracy:0.5910,  val_binary_crossentropy:0.6256,  val_loss:0.6256,  
....................................................................................................
Epoch: 200, accuracy:0.6269,  binary_crossentropy:0.6103,  loss:0.6103,  val_accuracy:0.6110,  val_binary_crossentropy:0.6151,  val_loss:0.6151,  
....................................................................................................
Epoch: 300, accuracy:0.6484,  binary_crossentropy:0.5984,  loss:0.5984,  val_accuracy:0.6340,  val_binary_crossentropy:0.6038,  val_loss:0.6038,  
....................................................................................................
Epoch: 400, accuracy:0.6584,  binary_crossentropy:0.5905,  loss:0.5905,  val_accuracy:0.6340,  val_binary_crossentropy:0.5993,  val_loss:0.5993,  
....................................................................................................
Epoch: 500, accuracy:0.6694,  binary_crossentropy:0.5860,  loss:0.5860,  val_accuracy:0.6410,  val_binary_crossentropy:0.5979,  val_loss:0.5979,  
....................................................................................................
Epoch: 600, accuracy:0.6684,  binary_crossentropy:0.5831,  loss:0.5831,  val_accuracy:0.6550,  val_binary_crossentropy:0.5960,  val_loss:0.5960,  
....................................................................................................
Epoch: 700, accuracy:0.6748,  binary_crossentropy:0.5810,  loss:0.5810,  val_accuracy:0.6510,  val_binary_crossentropy:0.5967,  val_loss:0.5967,  
....................................................................................................
Epoch: 800, accuracy:0.6707,  binary_crossentropy:0.5795,  loss:0.5795,  val_accuracy:0.6580,  val_binary_crossentropy:0.5965,  val_loss:0.5965,  
...........................
```

Now check how the model did:

```python
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
(0.5, 0.7)
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_dkEvb2x5XsjE_1.png)

### Small model

To check if you can beat the performance of the small model, progressively train some larger models.

Try two hidden layers with 16 units each:

```python
small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_2 (Dense)             (None, 16)                464       
                                                                 
 dense_3 (Dense)             (None, 16)                272       
                                                                 
 dense_4 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 753
Trainable params: 753
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.4877,  binary_crossentropy:0.7209,  loss:0.7209,  val_accuracy:0.4860,  val_binary_crossentropy:0.7025,  val_loss:0.7025,  
....................................................................................................
Epoch: 100, accuracy:0.6212,  binary_crossentropy:0.6148,  loss:0.6148,  val_accuracy:0.6200,  val_binary_crossentropy:0.6184,  val_loss:0.6184,  
....................................................................................................
Epoch: 200, accuracy:0.6657,  binary_crossentropy:0.5853,  loss:0.5853,  val_accuracy:0.6570,  val_binary_crossentropy:0.5949,  val_loss:0.5949,  
....................................................................................................
Epoch: 300, accuracy:0.6774,  binary_crossentropy:0.5750,  loss:0.5750,  val_accuracy:0.6720,  val_binary_crossentropy:0.5868,  val_loss:0.5868,  
....................................................................................................
Epoch: 400, accuracy:0.6838,  binary_crossentropy:0.5683,  loss:0.5683,  val_accuracy:0.6760,  val_binary_crossentropy:0.5859,  val_loss:0.5859,  
....................................................................................................
Epoch: 500, accuracy:0.6897,  binary_crossentropy:0.5632,  loss:0.5632,  val_accuracy:0.6720,  val_binary_crossentropy:0.5863,  val_loss:0.5863,  
....................................................................................................
Epoch: 600, accuracy:0.6946,  binary_crossentropy:0.5593,  loss:0.5593,  val_accuracy:0.6670,  val_binary_crossentropy:0.5883,  val_loss:0.5883,  
....................................................................................................
Epoch: 700, accuracy:0.6963,  binary_crossentropy:0.5558,  loss:0.5558,  val_accuracy:0.6730,  val_binary_crossentropy:0.5869,  val_loss:0.5869,  
....................................................................................................
Epoch: 800, accuracy:0.7006,  binary_crossentropy:0.5531,  loss:0.5531,  val_accuracy:0.6620,  val_binary_crossentropy:0.5894,  val_loss:0.5894,  
.........................
```

### Medium model

Now try three hidden layers with 64 units each:

```python
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])
```

And train the model using the same data:

```python
size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_5 (Dense)             (None, 64)                1856      
                                                                 
 dense_6 (Dense)             (None, 64)                4160      
                                                                 
 dense_7 (Dense)             (None, 64)                4160      
                                                                 
 dense_8 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 10,241
Trainable params: 10,241
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.4897,  binary_crossentropy:0.6952,  loss:0.6952,  val_accuracy:0.4970,  val_binary_crossentropy:0.6829,  val_loss:0.6829,  
....................................................................................................
Epoch: 100, accuracy:0.7220,  binary_crossentropy:0.5194,  loss:0.5194,  val_accuracy:0.6450,  val_binary_crossentropy:0.6157,  val_loss:0.6157,  
....................................................................................................
Epoch: 200, accuracy:0.7929,  binary_crossentropy:0.4219,  loss:0.4219,  val_accuracy:0.6490,  val_binary_crossentropy:0.7006,  val_loss:0.7006,  
.......................................................
```

### Large model

As an exercise, you can create an even larger model and check how quickly it begins overfitting. Next, add to this benchmark a network that has much more capacity, far more than the problem would warrant:

```python
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
```

And, again, train the model using the same data:

```python
size_histories['large'] = compile_and_fit(large_model, "sizes/large")
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_9 (Dense)             (None, 512)               14848     
                                                                 
 dense_10 (Dense)            (None, 512)               262656    
                                                                 
 dense_11 (Dense)            (None, 512)               262656    
                                                                 
 dense_12 (Dense)            (None, 512)               262656    
                                                                 
 dense_13 (Dense)            (None, 1)                 513       
                                                                 
=================================================================
Total params: 803,329
Trainable params: 803,329
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.5121,  binary_crossentropy:0.8038,  loss:0.8038,  val_accuracy:0.4670,  val_binary_crossentropy:0.6990,  val_loss:0.6990,  
....................................................................................................
Epoch: 100, accuracy:1.0000,  binary_crossentropy:0.0021,  loss:0.0021,  val_accuracy:0.6500,  val_binary_crossentropy:1.8178,  val_loss:1.8178,  
....................................................................................................
Epoch: 200, accuracy:1.0000,  binary_crossentropy:0.0001,  loss:0.0001,  val_accuracy:0.6590,  val_binary_crossentropy:2.4547,  val_loss:2.4547,  
......................
```

### Plot the training and validation losses

The solid lines show the training loss, and the dashed lines show the validation loss (remember: a lower validation loss indicates a better model).

While building a larger model gives it more power, if this power is not constrained somehow it can easily overfit to the training set.

In this example, typically, only the `"Tiny"` model manages to avoid overfitting altogether, and each of the larger models overfit the data more quickly. This becomes so severe for the `"large"` model that you need to switch the plot to a log-scale to really figure out what's happening.

This is apparent if you plot and compare the validation metrics to the training metrics.

- It's normal for there to be a small difference.
- If both metrics are moving in the same direction, everything is fine.
- If the validation metric begins to stagnate while the training metric continues to improve, you are probably close to overfitting.
- If the validation metric is going in the wrong direction, the model is clearly overfitting.

```python
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
Text(0.5, 0, 'Epochs [Log Scale]')
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_0XmKDtOWzOpk_1.png)

**Note:** All the above training runs used the [`callbacks.EarlyStopping`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) to end the training once it was clear the model was not making progress.

### View in TensorBoard

These models all wrote TensorBoard logs during training.

Open an embedded TensorBoard viewer inside a notebook:

```python
#docs_infra: no_execute

# Load the TensorBoard notebook extension
%load_ext tensorboard

# Open an embedded TensorBoard viewer
%tensorboard --logdir {logdir}/sizes
```

You can view the [results of a previous run](https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97) of this notebook on [TensorBoard.dev](https://tensorboard.dev/).

TensorBoard.dev is a managed experience for hosting, tracking, and sharing ML experiments with everyone.

It's also included in an `<iframe>` for convenience:

```python
display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px")
```



If you want to share TensorBoard results you can upload the logs to [TensorBoard.dev](https://tensorboard.dev/) by copying the following into a code-cell.

**Note:** This step requires a Google account.

```bsh
tensorboard dev upload --logdir  {logdir}/sizes
```

**Caution:** This command does not terminate. It's designed to continuously upload the results of long-running experiments. Once your data is uploaded you need to stop it using the "interrupt execution" option in your notebook tool.

## Strategies to prevent overfitting

Before getting into the content of this section copy the training logs from the `"Tiny"` model above, to use as a baseline for comparison.

```python
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')
PosixPath('/tmp/tmp0j45s09l/tensorboard_logs/regularizers/Tiny')
regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']
```

### Add weight regularization

You may be familiar with Occam's Razor principle: given two explanations for something, the explanation most likely to be correct is the "simplest" one, the one that makes the least amount of assumptions. This also applies to the models learned by neural networks: given some training data and a network architecture, there are multiple sets of weights values (multiple models) that could explain the data, and simpler models are less likely to overfit than complex ones.

A "simple model" in this context is a model where the distribution of parameter values has less entropy (or a model with fewer parameters altogether, as demonstrated in the section above). Thus a common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights only to take small values, which makes the distribution of weight values more "regular". This is called "weight regularization", and it is done by adding to the loss function of the network a cost associated with having large weights. This cost comes in two flavors:

- [L1 regularization](https://developers.google.com/machine-learning/glossary/#L1_regularization), where the cost added is proportional to the absolute value of the weights coefficients (i.e. to what is called the "L1 norm" of the weights).
- [L2 regularization](https://developers.google.com/machine-learning/glossary/#L2_regularization), where the cost added is proportional to the square of the value of the weights coefficients (i.e. to what is called the squared "L2 norm" of the weights). L2 regularization is also called weight decay in the context of neural networks. Don't let the different name confuse you: weight decay is mathematically the exact same as L2 regularization.

L1 regularization pushes weights towards exactly zero, encouraging a sparse model. L2 regularization will penalize the weights parameters without making them sparse since the penalty goes to zero for small weights—one reason why L2 is more common.

In [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras), weight regularization is added by passing weight regularizer instances to layers as keyword arguments. Add L2 weight regularization:

```python
l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_14 (Dense)            (None, 512)               14848     
                                                                 
 dense_15 (Dense)            (None, 512)               262656    
                                                                 
 dense_16 (Dense)            (None, 512)               262656    
                                                                 
 dense_17 (Dense)            (None, 512)               262656    
                                                                 
 dense_18 (Dense)            (None, 1)                 513       
                                                                 
=================================================================
Total params: 803,329
Trainable params: 803,329
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.5130,  binary_crossentropy:0.7964,  loss:2.3161,  val_accuracy:0.4900,  val_binary_crossentropy:0.6838,  val_loss:2.1253,  
....................................................................................................
Epoch: 100, accuracy:0.6464,  binary_crossentropy:0.6007,  loss:0.6227,  val_accuracy:0.6280,  val_binary_crossentropy:0.5894,  val_loss:0.6115,  
....................................................................................................
Epoch: 200, accuracy:0.6673,  binary_crossentropy:0.5865,  loss:0.6076,  val_accuracy:0.6830,  val_binary_crossentropy:0.5773,  val_loss:0.5984,  
....................................................................................................
Epoch: 300, accuracy:0.6767,  binary_crossentropy:0.5803,  loss:0.6010,  val_accuracy:0.6220,  val_binary_crossentropy:0.5999,  val_loss:0.6211,  
....................................................................................................
Epoch: 400, accuracy:0.6880,  binary_crossentropy:0.5677,  loss:0.5883,  val_accuracy:0.6750,  val_binary_crossentropy:0.5827,  val_loss:0.6033,  
............................................................................
```

`l2(0.001)` means that every coefficient in the weight matrix of the layer will add `0.001 * weight_coefficient_value**2` to the total **loss** of the network.

That is why we're monitoring the `binary_crossentropy` directly. Because it doesn't have this regularization component mixed in.

So, that same `"Large"` model with an `L2` regularization penalty performs much better:

```python
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
(0.5, 0.7)
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_7wkfLyxBZdh__1.png)

As demonstrated in the diagram above, the `"L2"` regularized model is now much more competitive with the `"Tiny"` model. This `"L2"` model is also much more resistant to overfitting than the `"Large"` model it was based on despite having the same number of parameters.

#### More info

There are two important things to note about this sort of regularization:

1. If you are writing your own training loop, then you need to be sure to ask the model for its regularization losses.

```python
result = l2_model(features)
regularization_loss=tf.add_n(l2_model.losses)
```

1. This implementation works by adding the weight penalties to the model's loss, and then applying a standard optimization procedure after that.

There is a second approach that instead only runs the optimizer on the raw loss, and then while applying the calculated step the optimizer also applies some weight decay. This "decoupled weight decay" is used in optimizers like [`tf.keras.optimizers.Ftrl`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl) and [`tfa.optimizers.AdamW`](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW).

### Add dropout

Dropout is one of the most effective and most commonly used regularization techniques for neural networks, developed by Hinton and his students at the University of Toronto.

The intuitive explanation for dropout is that because individual nodes in the network cannot rely on the output of the others, each node must output features that are useful on their own.

Dropout, applied to a layer, consists of randomly "dropping out" (i.e. set to zero) a number of output features of the layer during training. For example, a given layer would normally have returned a vector `[0.2, 0.5, 1.3, 0.8, 1.1]` for a given input sample during training; after applying dropout, this vector will have a few zero entries distributed at random, e.g. `[0, 0.5, 1.3, 0, 1.1]`.

The "dropout rate" is the fraction of the features that are being zeroed-out; it is usually set between 0.2 and 0.5. At test time, no units are dropped out, and instead the layer's output values are scaled down by a factor equal to the dropout rate, so as to balance for the fact that more units are active than at training time.

In Keras, you can introduce dropout in a network via the [`tf.keras.layers.Dropout`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) layer, which gets applied to the output of layer right before.

Add two dropout layers to your network to check how well they do at reducing overfitting:

```python
dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_19 (Dense)            (None, 512)               14848     
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_20 (Dense)            (None, 512)               262656    
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_21 (Dense)            (None, 512)               262656    
                                                                 
 dropout_2 (Dropout)         (None, 512)               0         
                                                                 
 dense_22 (Dense)            (None, 512)               262656    
                                                                 
 dropout_3 (Dropout)         (None, 512)               0         
                                                                 
 dense_23 (Dense)            (None, 1)                 513       
                                                                 
=================================================================
Total params: 803,329
Trainable params: 803,329
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.5111,  binary_crossentropy:0.7826,  loss:0.7826,  val_accuracy:0.5150,  val_binary_crossentropy:0.6853,  val_loss:0.6853,  
....................................................................................................
Epoch: 100, accuracy:0.6559,  binary_crossentropy:0.5941,  loss:0.5941,  val_accuracy:0.6960,  val_binary_crossentropy:0.5853,  val_loss:0.5853,  
....................................................................................................
Epoch: 200, accuracy:0.6962,  binary_crossentropy:0.5544,  loss:0.5544,  val_accuracy:0.6870,  val_binary_crossentropy:0.5878,  val_loss:0.5878,  
....................................................................................................
Epoch: 300, accuracy:0.7242,  binary_crossentropy:0.5066,  loss:0.5066,  val_accuracy:0.6770,  val_binary_crossentropy:0.6016,  val_loss:0.6016,  
..........................................................
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
(0.5, 0.7)
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_SPZqwVchx5xp_1.png)

It's clear from this plot that both of these regularization approaches improve the behavior of the `"Large"` model. But this still doesn't beat even the `"Tiny"` baseline.

Next try them both, together, and see if that does better.

### Combined L2 + dropout

```python
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")
Model: "sequential_6"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_24 (Dense)            (None, 512)               14848     
                                                                 
 dropout_4 (Dropout)         (None, 512)               0         
                                                                 
 dense_25 (Dense)            (None, 512)               262656    
                                                                 
 dropout_5 (Dropout)         (None, 512)               0         
                                                                 
 dense_26 (Dense)            (None, 512)               262656    
                                                                 
 dropout_6 (Dropout)         (None, 512)               0         
                                                                 
 dense_27 (Dense)            (None, 512)               262656    
                                                                 
 dropout_7 (Dropout)         (None, 512)               0         
                                                                 
 dense_28 (Dense)            (None, 1)                 513       
                                                                 
=================================================================
Total params: 803,329
Trainable params: 803,329
Non-trainable params: 0
_________________________________________________________________

Epoch: 0, accuracy:0.5009,  binary_crossentropy:0.8059,  loss:0.9642,  val_accuracy:0.4610,  val_binary_crossentropy:0.7068,  val_loss:0.8645,  
....................................................................................................
Epoch: 100, accuracy:0.6455,  binary_crossentropy:0.6040,  loss:0.6345,  val_accuracy:0.6240,  val_binary_crossentropy:0.5893,  val_loss:0.6197,  
....................................................................................................
Epoch: 200, accuracy:0.6583,  binary_crossentropy:0.5897,  loss:0.6156,  val_accuracy:0.6720,  val_binary_crossentropy:0.5737,  val_loss:0.5996,  
....................................................................................................
Epoch: 300, accuracy:0.6743,  binary_crossentropy:0.5808,  loss:0.6092,  val_accuracy:0.6860,  val_binary_crossentropy:0.5623,  val_loss:0.5907,  
....................................................................................................
Epoch: 400, accuracy:0.6748,  binary_crossentropy:0.5779,  loss:0.6084,  val_accuracy:0.6970,  val_binary_crossentropy:0.5520,  val_loss:0.5825,  
....................................................................................................
Epoch: 500, accuracy:0.6843,  binary_crossentropy:0.5704,  loss:0.6033,  val_accuracy:0.6760,  val_binary_crossentropy:0.5598,  val_loss:0.5927,  
....................................................................................................
Epoch: 600, accuracy:0.6868,  binary_crossentropy:0.5679,  loss:0.6028,  val_accuracy:0.6880,  val_binary_crossentropy:0.5448,  val_loss:0.5796,  
....................................................................................................
Epoch: 700, accuracy:0.6944,  binary_crossentropy:0.5599,  loss:0.5966,  val_accuracy:0.6930,  val_binary_crossentropy:0.5480,  val_loss:0.5847,  
....................................................................................................
Epoch: 800, accuracy:0.6891,  binary_crossentropy:0.5571,  loss:0.5951,  val_accuracy:0.6950,  val_binary_crossentropy:0.5466,  val_loss:0.5846,  
......................................
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
(0.5, 0.7)
```

![png](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit_files/output_qDqBBxfI0Yd8_1.png)

This model with the `"Combined"` regularization is obviously the best one so far.

### View in TensorBoard

These models also recorded TensorBoard logs.

To open an embedded tensorboard viewer inside a notebook, copy the following into a code-cell:

```
%tensorboard --logdir {logdir}/regularizers
```

You can view the [results of a previous run](https://tensorboard.dev/experiment/fGInKDo8TXes1z7HQku9mw/#scalars&_smoothingWeight=0.97) of this notebook on [TensorBoard.dev](https://tensorboard.dev/).

It's also included in an `<iframe>` for convenience:

```python
display.IFrame(
    src="https://tensorboard.dev/experiment/fGInKDo8TXes1z7HQku9mw/#scalars&_smoothingWeight=0.97",
    width = "100%",
    height="800px")
```



This was uploaded with:

```bsh
tensorboard dev upload --logdir  {logdir}/regularizers
```

## Conclusions

To recap, here are the most common ways to prevent overfitting in neural networks:

- Get more training data.
- Reduce the capacity of the network.
- Add weight regularization.
- Add dropout.

Two important approaches not covered in this guide are:

- [Data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
- Batch normalization ([`tf.keras.layers.BatchNormalization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization))

Remember that each method can help on its own, but often combining them can be even more effective.