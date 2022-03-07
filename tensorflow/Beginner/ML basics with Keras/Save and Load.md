## Save and load



可以在训练期间和之后保存模型进度。这意味着模型可以从停止的地方恢复，避免长时间的训练。此外，保存还意味着您可以分享您的模型，其他人可以重现您的工作。在发布研究模型和技术时，大多数机器学习从业者会分享：

- 用于创建模型的代码
- 模型训练的权重 (weight) 和参数 (parameters) 。

共享数据有助于其他人了解模型的工作原理，并使用新数据自行尝试。

小心：TensorFlow 模型是代码，对于不受信任的代码，一定要小心。请参阅 [安全使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)以了解详情。

### 选项

根据您使用的 API，可以通过多种方式保存 TensorFlow 模型。本指南使用 [tf.keras](https://tensorflow.google.cn/guide/keras)，这是一种在 TensorFlow 中构建和训练模型的高级 API。对于其他方式，请参阅 TensorFlow [保存和恢复](https://tensorflow.google.cn/guide/saved_model)指南或[在 Eager 中保存](https://tensorflow.google.cn/guide/eager#object-based_saving)。



## 配置

### 安装并导入

安装并导入Tensorflow和依赖项：

```bsh
pip install pyyaml h5py  # Required to save models in HDF5 format
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)
2.6.0
```

### 获取示例数据集

为了演示如何保存和加载权重，您将使用 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)。为了加快运行速度，请使用前 1000 个样本：

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```

### 定义模型

首先构建一个简单的序列（sequential）模型：

```python
# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
2021-08-13 23:43:58.306647: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
2021-08-13 23:43:58.314878: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.315788: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.317873: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-08-13 23:43:58.318529: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.319522: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.320401: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.937765: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.938713: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.939555: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:43:58.940373: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14648 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:05.0, compute capability: 7.0
```

## 在训练期间保存模型（以 checkpoints 形式保存）

您可以使用经过训练的模型而无需重新训练，或者在训练过程中断的情况下从离开处继续训练。`tf.keras.callbacks.ModelCheckpoint` 回调允许您在训练*期间*和*结束*时持续保存模型。

### Checkpoint 回调用法

创建一个只在训练期间保存权重的 `tf.keras.callbacks.ModelCheckpoint` 回调：

```python
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.
2021-08-13 23:43:59.457606: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/10
32/32 [==============================] - 1s 8ms/step - loss: 1.1507 - sparse_categorical_accuracy: 0.6640 - val_loss: 0.6974 - val_sparse_categorical_accuracy: 0.7860

Epoch 00001: saving model to training_1/cp.ckpt
Epoch 2/10
32/32 [==============================] - 0s 4ms/step - loss: 0.4174 - sparse_categorical_accuracy: 0.8840 - val_loss: 0.5499 - val_sparse_categorical_accuracy: 0.8310

Epoch 00002: saving model to training_1/cp.ckpt
Epoch 3/10
32/32 [==============================] - 0s 4ms/step - loss: 0.2882 - sparse_categorical_accuracy: 0.9240 - val_loss: 0.5217 - val_sparse_categorical_accuracy: 0.8360

Epoch 00003: saving model to training_1/cp.ckpt
Epoch 4/10
32/32 [==============================] - 0s 4ms/step - loss: 0.2154 - sparse_categorical_accuracy: 0.9500 - val_loss: 0.4670 - val_sparse_categorical_accuracy: 0.8450

Epoch 00004: saving model to training_1/cp.ckpt
Epoch 5/10
32/32 [==============================] - 0s 4ms/step - loss: 0.1484 - sparse_categorical_accuracy: 0.9700 - val_loss: 0.4139 - val_sparse_categorical_accuracy: 0.8710

Epoch 00005: saving model to training_1/cp.ckpt
Epoch 6/10
32/32 [==============================] - 0s 4ms/step - loss: 0.1245 - sparse_categorical_accuracy: 0.9760 - val_loss: 0.4361 - val_sparse_categorical_accuracy: 0.8480

Epoch 00006: saving model to training_1/cp.ckpt
Epoch 7/10
32/32 [==============================] - 0s 4ms/step - loss: 0.0905 - sparse_categorical_accuracy: 0.9860 - val_loss: 0.4034 - val_sparse_categorical_accuracy: 0.8640

Epoch 00007: saving model to training_1/cp.ckpt
Epoch 8/10
32/32 [==============================] - 0s 4ms/step - loss: 0.0668 - sparse_categorical_accuracy: 0.9940 - val_loss: 0.4261 - val_sparse_categorical_accuracy: 0.8640

Epoch 00008: saving model to training_1/cp.ckpt
Epoch 9/10
32/32 [==============================] - 0s 4ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9950 - val_loss: 0.4191 - val_sparse_categorical_accuracy: 0.8640

Epoch 00009: saving model to training_1/cp.ckpt
Epoch 10/10
32/32 [==============================] - 0s 4ms/step - loss: 0.0377 - sparse_categorical_accuracy: 0.9970 - val_loss: 0.4329 - val_sparse_categorical_accuracy: 0.8640

Epoch 00010: saving model to training_1/cp.ckpt
<keras.callbacks.History at 0x7fa9cc0f0350>
```

这将创建一个 TensorFlow checkpoint 文件集合，这些文件在每个 epoch 结束时更新：

```python
os.listdir(checkpoint_dir)
['cp.ckpt.index', 'cp.ckpt.data-00000-of-00001', 'checkpoint']
```

只要两个模型共享相同的架构，您就可以在它们之间共享权重。因此，当从仅权重恢复模型时，创建一个与原始模型具有相同架构的模型，然后设置其权重。

现在，重新构建一个未经训练的全新模型并基于测试集对其进行评估。未经训练的模型将以机会水平执行（约 10% 的准确率）：

```python
# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
32/32 - 0s - loss: 2.3609 - sparse_categorical_accuracy: 0.1150
Untrained model, accuracy: 11.50%
```

然后从 checkpoint 加载权重并重新评估：

```python
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
32/32 - 0s - loss: 0.4329 - sparse_categorical_accuracy: 0.8640
Restored model, accuracy: 86.40%
```

### checkpoint 回调选项

回调提供了几个选项，为 checkpoint 提供唯一名称并调整 checkpoint 频率。

训练一个新模型，每五个 epochs 保存一次唯一命名的 checkpoint ：

```python
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)
Epoch 00005: saving model to training_2/cp-0005.ckpt

Epoch 00010: saving model to training_2/cp-0010.ckpt

Epoch 00015: saving model to training_2/cp-0015.ckpt

Epoch 00020: saving model to training_2/cp-0020.ckpt

Epoch 00025: saving model to training_2/cp-0025.ckpt

Epoch 00030: saving model to training_2/cp-0030.ckpt

Epoch 00035: saving model to training_2/cp-0035.ckpt

Epoch 00040: saving model to training_2/cp-0040.ckpt

Epoch 00045: saving model to training_2/cp-0045.ckpt

Epoch 00050: saving model to training_2/cp-0050.ckpt
<keras.callbacks.History at 0x7fa97679a950>
```

现在查看生成的 checkpoint 并选择最新的 checkpoint ：

```python
os.listdir(checkpoint_dir)
['cp-0030.ckpt.index',
 'cp-0015.ckpt.index',
 'cp-0000.ckpt.data-00000-of-00001',
 'cp-0005.ckpt.index',
 'cp-0010.ckpt.index',
 'cp-0020.ckpt.data-00000-of-00001',
 'cp-0040.ckpt.data-00000-of-00001',
 'cp-0015.ckpt.data-00000-of-00001',
 'cp-0025.ckpt.data-00000-of-00001',
 'cp-0025.ckpt.index',
 'cp-0050.ckpt.index',
 'cp-0050.ckpt.data-00000-of-00001',
 'cp-0035.ckpt.index',
 'cp-0020.ckpt.index',
 'cp-0030.ckpt.data-00000-of-00001',
 'cp-0045.ckpt.index',
 'cp-0000.ckpt.index',
 'cp-0010.ckpt.data-00000-of-00001',
 'cp-0035.ckpt.data-00000-of-00001',
 'cp-0040.ckpt.index',
 'cp-0045.ckpt.data-00000-of-00001',
 'checkpoint',
 'cp-0005.ckpt.data-00000-of-00001']
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
'training_2/cp-0050.ckpt'
```

注：默认 TensorFlow 格式只保存最近的 5 个检查点。

如果要进行测试，请重置模型并加载最新的 checkpoint ：

```python
# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
32/32 - 0s - loss: 0.4978 - sparse_categorical_accuracy: 0.8750
Restored model, accuracy: 87.50%
```

## 这些文件是什么？

上述代码将权重存储到 [checkpoint](https://tensorflow.google.cn/guide/saved_model#save_and_restore_variables)—— 格式化文件的集合中，这些文件仅包含二进制格式的训练权重。 Checkpoints 包含：

- 一个或多个包含模型权重的分片。
- 一个索引文件，指示哪些权重存储在哪个分片中。

如果您在一台计算机上训练模型，您将获得一个具有如下后缀的分片：`.data-00000-of-00001`

## 手动保存权重

使用 `Model.save_weights` 方法手动保存权重。默认情况下，`tf.keras`（尤其是 `save_weights`）使用扩展名为 `.ckpt` 的 TensorFlow [检查点](https://www.tensorflow.org/guide/checkpoint)格式（保存在扩展名为 `.h5` 的 [HDF5](https://js.tensorflow.org/tutorials/import-keras.html)中，[保存和序列化模型](https://www.tensorflow.org/guide/keras/save_and_serialize#weights-only_saving_in_savedmodel_format)指南中会讲到这一点）：

```python
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.iter
WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1
WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_2
WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.decay
WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.learning_rate
WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
32/32 - 0s - loss: 0.4978 - sparse_categorical_accuracy: 0.8750
Restored model, accuracy: 87.50%
```

## 保存整个模型

调用 [`model.save`](https://tensorflow.google.cn/api_docs/python/tf/keras/Model#save) 将保存模型的结构，权重和训练配置保存在单个文件/文件夹中。这可以让您导出模型，以便在不访问原始 Python 代码*的情况下使用它。因为优化器状态（optimizer-state）已经恢复，您可以从中断的位置恢复训练。

整个模型可以保存为两种不同的文件格式（`SavedModel` 和 `HDF5`）。TensorFlow `SavedModel` 格式是 TF2.x 中的默认文件格式。但是，模型能够以 `HDF5` 格式保存。下面详细介绍了如何以两种文件格式保存整个模型。

保存完整模型会非常有用——您可以在 TensorFlow.js（[Saved Model](https://tensorflow.google.cn/js/tutorials/conversion/import_saved_model), [HDF5](https://tensorflow.google.cn/js/tutorials/conversion/import_keras)）加载它们，然后在 web 浏览器中训练和运行它们，或者使用 TensorFlow Lite 将它们转换为在移动设备上运行（[Saved Model](https://tensorflow.google.cn/lite/convert/python_api#converting_a_savedmodel_), [HDF5](https://tensorflow.google.cn/lite/convert/python_api#converting_a_keras_model_)）

*自定义对象（例如，子类化模型或层）在保存和加载时需要特别注意。请参阅下面的**保存自定义对象**部分 

### SavedModel 格式

SavedModel 格式是另一种序列化模型的方式。以这种格式保存的模型可以使用 `tf.keras.models.load_model` 恢复，并且与 TensorFlow Serving 兼容。[SavedModel 指南](https://tensorflow.google.cn/guide/saved_model)详细介绍了如何应用/检查 SavedModel。以下部分说明了保存和恢复模型的步骤。

```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')
Epoch 1/5
32/32 [==============================] - 0s 2ms/step - loss: 1.1543 - sparse_categorical_accuracy: 0.6700
Epoch 2/5
32/32 [==============================] - 0s 2ms/step - loss: 0.4327 - sparse_categorical_accuracy: 0.8830
Epoch 3/5
32/32 [==============================] - 0s 2ms/step - loss: 0.2942 - sparse_categorical_accuracy: 0.9220
Epoch 4/5
32/32 [==============================] - 0s 2ms/step - loss: 0.2200 - sparse_categorical_accuracy: 0.9460
Epoch 5/5
32/32 [==============================] - 0s 2ms/step - loss: 0.1600 - sparse_categorical_accuracy: 0.9650
2021-08-13 23:44:09.574998: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
INFO:tensorflow:Assets written to: saved_model/my_model/assets
```

SavedModel 格式是一个包含 protobuf 二进制文件和 TensorFlow 检查点的目录。检查保存的模型目录：

```bsh
# my_model directory
ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
ls saved_model/my_model
my_model
assets  keras_metadata.pb  saved_model.pb  variables
```

从保存的模型重新加载一个新的 Keras 模型：

```python
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_10 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_5 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

使用与原始模型相同的参数编译恢复的模型。尝试使用加载的模型运行评估和预测：

```python
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)
32/32 - 0s - loss: 0.4178 - sparse_categorical_accuracy: 0.8690
Restored model, accuracy: 86.90%
(1000, 10)
```

### HDF5 格式

Keras使用 [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) 标准提供了一种基本的保存格式。 

```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')
Epoch 1/5
32/32 [==============================] - 0s 2ms/step - loss: 1.1866 - sparse_categorical_accuracy: 0.6520
Epoch 2/5
32/32 [==============================] - 0s 2ms/step - loss: 0.4286 - sparse_categorical_accuracy: 0.8820
Epoch 3/5
32/32 [==============================] - 0s 2ms/step - loss: 0.2841 - sparse_categorical_accuracy: 0.9280
Epoch 4/5
32/32 [==============================] - 0s 2ms/step - loss: 0.2033 - sparse_categorical_accuracy: 0.9560
Epoch 5/5
32/32 [==============================] - 0s 2ms/step - loss: 0.1580 - sparse_categorical_accuracy: 0.9630
```

现在，从该文件重新创建模型：

```python
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_12 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_13 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

检查其准确率（accuracy）：

```python
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
32/32 - 0s - loss: 0.4283 - sparse_categorical_accuracy: 0.8600
Restored model, accuracy: 86.00%
```

Keras 通过检查模型的架构来保存这些模型。这种技术可以保存所有内容：

- 权重值
- 模型的架构
- 模型的训练配置（您传递给 `.compile()` 方法的内容）
- 优化器及其状态（如果有）（这样，您便可从中断的地方重新启动训练）

Keras 无法保存 `v1.x` 优化器（来自 `tf.compat.v1.train`），因为它们与检查点不兼容。对于 v1.x 优化器，您需要在加载-失去优化器的状态后，重新编译模型。

### Saving custom objects

If you are using the SavedModel format, you can skip this section. The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture, while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and custom layers without requiring the original code.

To save custom objects to HDF5, you must do the following:

1. Define a `get_config`method in your object, and optionally a `from_config`classmethod.

   - `get_config(self)` returns a JSON-serializable dictionary of parameters needed to recreate the object.
   - `from_config(cls, config)` uses the returned config from `get_config` to create a new object. By default, this function will use the config as initialization kwargs (`return cls(**config)`).
   
2. Pass the object to the `custom_objects` argument when loading the model. The argument must be a dictionary mapping the string class name to the Python class. E.g. `tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})`

See the [Writing layers and models from scratch](https://www.tensorflow.org/guide/keras/custom_layers_and_models) tutorial for examples of custom objects and `get_config`.

