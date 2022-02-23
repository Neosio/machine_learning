# Basic regression: Predict fuel efficiency

在 *回归 (regression)* 问题中，我们的目的是预测出如价格或概率这样连续值的输出。相对于*分类(classification)* 问题，*分类(classification)* 的目的是从一系列的分类出选择出一个分类 （如，给出一张包含苹果或橘子的图片，识别出图片中是哪种水果）。

本 notebook 使用经典的 [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) 数据集，构建了一个用来预测70年代末到80年代初汽车燃油效率的模型。为了做到这一点，我们将为该模型提供许多那个时期的汽车描述。这个描述包含：气缸数，排量，马力以及重量。

This example uses the Keras API. (Visit the Keras [tutorials](https://www.tensorflow.org/tutorials/keras) and [guides](https://www.tensorflow.org/guide/keras) to learn more.)

```bsh
# Use seaborn for pairplot.
pip install -q seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
2.8.0
```

## The Auto MPG dataset

The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/).

### Get the data

First download and import the dataset using pandas:

```python
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.tail()
```



### Clean the data

The dataset contains a few unknown values:

```python
dataset.isna().sum()
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
```

Drop those rows to keep this initial tutorial simple:

```python
dataset = dataset.dropna()
```

The `"Origin"` column is categorical, not numeric. So the next step is to one-hot encode the values in the column with [pd.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).

**Note:** You can set up the [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) to do this kind of transformation for you but that's beyond the scope of this tutorial. Check out the [Classify structured data using Keras preprocessing layers](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers) or [Load CSV data](https://www.tensorflow.org/tutorials/load_data/csv) tutorials for examples.

```python
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()
```



### Split the data into training and test sets

Now, split the dataset into a training set and a test set. You will use the test set in the final evaluation of your models.

```python
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
```

### Inspect the data

Review the joint distribution of a few pairs of columns from the training set.

The top row suggests that the fuel efficiency (MPG) is a function of all the other parameters. The other rows indicate they are functions of each other.

```python
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
<seaborn.axisgrid.PairGrid at 0x7f3f8ac88b50>
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_oRKO_x8gWKv-_1.png)

Let's also check the overall statistics. Note how each feature covers a very different range:

```python
train_dataset.describe().transpose()
```



### Split features from labels

Separate the target value—the "label"—from the features. This label is the value that you will train the model to predict.

```python
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')
```

## Normalization

In the table of statistics it's easy to see how different the ranges of each feature are:

```python
train_dataset.describe().transpose()[['mean', 'std']]
```



It is good practice to normalize features that use different scales and ranges.

One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.

Although a model *might* converge without feature normalization, normalization makes training much more stable.

**Note:** There is no advantage to normalizing the one-hot features—it is done here for simplicity. For more details on how to use the preprocessing layers, refer to the [Working with preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) guide and the [Classify structured data using Keras preprocessing layers](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers) tutorial.

### The Normalization layer

The [`tf.keras.layers.Normalization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization) is a clean and simple way to add feature normalization into your model.

The first step is to create the layer:

```python
normalizer = tf.keras.layers.Normalization(axis=-1)
```

Then, fit the state of the preprocessing layer to the data by calling [`Normalization.adapt`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization#adapt):

```python
normalizer.adapt(np.array(train_features))
```

Calculate the mean and variance, and store them in the layer:

```python
print(normalizer.mean.numpy())
[[   5.478  195.318  104.869 2990.252   15.559   75.898    0.178    0.197
     0.624]]
```

When the layer is called, it returns the input data, with each feature independently normalized:

```python
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
First example: [[   4.    90.    75.  2125.    14.5   74.     0.     0.     1. ]]

Normalized: [[-0.87 -1.01 -0.79 -1.03 -0.38 -0.52 -0.47 -0.5   0.78]]
```

## Linear regression

Before building a deep neural network model, start with linear regression using one and several variables.

### Linear regression with one variable

Begin with a single-variable linear regression to predict `'MPG'` from `'Horsepower'`.

Training a model with [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) typically starts by defining the model architecture. Use a [`tf.keras.Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) model, which [represents a sequence of steps](https://www.tensorflow.org/guide/keras/sequential_model).

There are two steps in your single-variable linear regression model:

- Normalize the `'Horsepower'` input features using the `tf.keras.layers.Normalization`preprocessing layer.
- Apply a linear transformation () to produce 1 output using a linear layer ([`tf.keras.layers.Dense`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)).

The number of *inputs* can either be set by the `input_shape` argument, or automatically when the model is run for the first time.

First, create a NumPy array made of the `'Horsepower'` features. Then, instantiate the `tf.keras.layers.Normalization` and fit its state to the `horsepower` data:

```python
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)
```

Build the Keras Sequential model:

```python
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 normalization_1 (Normalizat  (None, 1)                3         
 ion)                                                            
                                                                 
 dense (Dense)               (None, 1)                 2         
                                                                 
=================================================================
Total params: 5
Trainable params: 2
Non-trainable params: 3
_________________________________________________________________
```

This model will predict `'MPG'` from `'Horsepower'`.

Run the untrained model on the first 10 'Horsepower' values. The output won't be good, but notice that it has the expected shape of `(10, 1)`:

```python
horsepower_model.predict(horsepower[:10])
array([[-0.571],
       [-0.322],
       [ 1.054],
       [-0.8  ],
       [-0.724],
       [-0.284],
       [-0.858],
       [-0.724],
       [-0.189],
       [-0.322]], dtype=float32)
```

Once the model is built, configure the training procedure using the Keras [`Model.compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) method. The most important arguments to compile are the `loss` and the `optimizer`, since these define what will be optimized (`mean_absolute_error`) and how (using the [`tf.keras.optimizers.Adam`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)).

```python
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
```

Use Keras [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) to execute the training for 100 epochs:

```python
%%time
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
CPU times: user 4.88 s, sys: 809 ms, total: 5.69 s
Wall time: 3.82 s
```

Visualize the model's training progress using the stats stored in the `history` object:

```python
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
```



```python
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
plot_loss(history)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_yYsQYrIZyqjz_0.png)

Collect the results on the test set for later:

```python
test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)
```

Since this is a single variable regression, it's easy to view the model's predictions as a function of the input:

```python
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)
def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
plot_horsepower(x, y)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_7l9ZiAOEUNBL_0.png)

### Linear regression with multiple inputs

You can use an almost identical setup to make predictions based on multiple inputs. This model still does the same except that is a matrix and is a vector.

Create a two-step Keras Sequential model again with the first layer being `normalizer`(`tf.keras.layers.Normalization(axis=-1)`) you defined earlier and adapted to the whole dataset:

```python
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
```

When you call [`Model.predict`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict) on a batch of inputs, it produces `units=1` outputs for each example:

```python
linear_model.predict(train_features[:10])
array([[ 0.324],
       [-0.46 ],
       [-1.102],
       [ 0.466],
       [ 1.012],
       [-1.399],
       [ 0.817],
       [-0.906],
       [-1.116],
       [ 0.648]], dtype=float32)
```

When you call the model, its weight matrices will be built—check that the `kernel` weights (the in ) have a shape of `(9, 1)`:

```python
linear_model.layers[1].kernel
<tf.Variable 'dense_1/kernel:0' shape=(9, 1) dtype=float32, numpy=
array([[-0.148],
       [-0.714],
       [ 0.694],
       [-0.349],
       [-0.679],
       [ 0.235],
       [ 0.448],
       [ 0.546],
       [ 0.006]], dtype=float32)>
```

Configure the model with Keras [`Model.compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile) and train with [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) for 100 epochs:

```python
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
%%time
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
CPU times: user 4.54 s, sys: 799 ms, total: 5.34 s
Wall time: 3.52 s
```

Using all the inputs in this regression model achieves a much lower training and validation error than the `horsepower_model`, which had one input:

```python
plot_loss(history)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_4sWO3W0koYgu_0.png)

Collect the results on the test set for later:

```python
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)
```

## Regression with a deep neural network (DNN)

In the previous section, you implemented two linear models for single and multiple inputs.

Here, you will implement single-input and multiple-input DNN models.

The code is basically the same except the model is expanded to include some "hidden" non-linear layers. The name "hidden" here just means not directly connected to the inputs or outputs.

These models will contain a few more layers than the linear model:

- The normalization layer, as before (with `horsepower_normalizer` for a single-input model and `normalizer` for a multiple-input model).
- Two hidden, non-linear, `Dense` layers with the ReLU (`relu`) activation function nonlinearity.
- A linear `Dense` single-output layer.

Both models will use the same training procedure so the `compile` method is included in the `build_and_compile_model` function below.

```python
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model
```

### Regression using a DNN and a single input

Create a DNN model with only `'Horsepower'` as input and `horsepower_normalizer` (defined earlier) as the normalization layer:

```python
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
```

This model has quite a few more trainable parameters than the linear models:

```python
dnn_horsepower_model.summary()
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 normalization_1 (Normalizat  (None, 1)                3         
 ion)                                                            
                                                                 
 dense_2 (Dense)             (None, 64)                128       
                                                                 
 dense_3 (Dense)             (None, 64)                4160      
                                                                 
 dense_4 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 4,356
Trainable params: 4,353
Non-trainable params: 3
_________________________________________________________________
```

Train the model with Keras [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit):

```python
%%time
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
CPU times: user 4.83 s, sys: 829 ms, total: 5.66 s
Wall time: 3.8 s
```

This model does slightly better than the linear single-input `horsepower_model`:

```python
plot_loss(history)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_NcF6UWjdCU8T_0.png)

If you plot the predictions as a function of `'Horsepower'`, you should notice how this model takes advantage of the nonlinearity provided by the hidden layers:

```python
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
plot_horsepower(x, y)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_rsf9rD8I17Wq_0.png)

Collect the results on the test set for later:

```python
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)
```

### Regression using a DNN and multiple inputs

Repeat the previous process using all the inputs. The model's performance slightly improves on the validation dataset.

```python
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 normalization (Normalizatio  (None, 9)                19        
 n)                                                              
                                                                 
 dense_5 (Dense)             (None, 64)                640       
                                                                 
 dense_6 (Dense)             (None, 64)                4160      
                                                                 
 dense_7 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 4,884
Trainable params: 4,865
Non-trainable params: 19
_________________________________________________________________
%%time
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
CPU times: user 5.06 s, sys: 652 ms, total: 5.71 s
Wall time: 3.82 s
plot_loss(history)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_-9Dbj0fX23RQ_0.png)

Collect the results on the test set:

```python
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
```

## Performance

Since all models have been trained, you can review their test set performance:

```python
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
```



These results match the validation error observed during training.

### Make predictions

You can now make predictions with the `dnn_model` on the test set using Keras [`Model.predict`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)and review the loss:

```python
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_Xe7RXH3N3CWU_0.png)

It appears that the model predicts reasonably well.

Now, check the error distribution:

```python
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
```

![png](https://www.tensorflow.org/tutorials/keras/regression_files/output_f-OHX4DiXd8x_0.png)

If you're happy with the model, save it for later use with [`Model.save`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save):

```python
dnn_model.save('dnn_model')
2022-02-10 18:50:01.821887: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
INFO:tensorflow:Assets written to: dnn_model/assets
```

If you reload the model, it gives identical output:

```python
reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
```



## Conclusion

This notebook introduced a few techniques to handle a regression problem. Here are a few more tips that may help:

- Mean squared error (MSE) ([`tf.keras.losses.MeanSquaredError`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError)) and mean absolute error (MAE) ([`tf.keras.losses.MeanAbsoluteError`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanAbsoluteError)) are common loss functions used for regression problems. MAE is less sensitive to outliers. Different loss functions are used for classification problems.
- Similarly, evaluation metrics used for regression differ from classification.
- When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.
- Overfitting is a common problem for DNN models, though it wasn't a problem for this tutorial. Visit the [Overfit and underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) tutorial for more help with this.