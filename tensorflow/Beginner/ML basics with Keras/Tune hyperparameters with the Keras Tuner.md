## 使用Keras Tuner调整超参数

## 概述

Keras Tuner 是一个库，可帮助您为 TensorFlow 程序选择最佳的超参数集。为您的机器学习 (ML) 应用选择正确的超参数集，这一过程称为*超参数调节*或*超调*。

超参数是控制训练过程和 ML 模型拓扑的变量。这些变量在训练过程中保持不变，并会直接影响 ML 程序的性能。超参数有两种类型：

1. **模型超参数**：影响模型的选择，例如隐藏层的数量和宽度
2. **算法超参数**：影响学习算法的速度和质量，例如随机梯度下降 (SGD) 的学习率以及 k 近邻 (KNN) 分类器的近邻数

在本教程中，您将使用 Keras Tuner 对图像分类应用执行超调。

## 设置

```python
import tensorflow as tf
from tensorflow import keras
```

安装并导入 Keras Tuner。

```bsh
pip install -q -U keras-tuner
import keras_tuner as kt
```

## 下载并准备数据集

在本教程中，您将使用 Keras Tuner 为某个对 [Fashion MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist)内的服装图像进行分类的机器学习模型找到最佳超参数。

加载数据。

```python
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0
```

## 定义模型

构建用于超调的模型时，除了模型架构之外，还要定义超参数搜索空间。您为超调设置的模型称为*超模型*。

您可以通过两种方式定义超模型：

- 使用模型构建工具函数
- 将 Keras Tuner API 的 `HyperModel` 类子类化

您还可以将两个预定义的 `HyperModel` 类（[HyperXception](https://keras-team.github.io/keras-tuner/documentation/hypermodels/#hyperxception-class) 和 [HyperResNet](https://keras-team.github.io/keras-tuner/documentation/hypermodels/#hyperresnet-class)）用于计算机视觉应用。

在本教程中，您将使用模型构建工具函数来定义图像分类模型。模型构建工具函数将返回已编译的模型，并使用您以内嵌方式定义的超参数对模型进行超调。

```python
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model
```

## 实例化调节器并执行超调

实例化调节器以执行超调。Keras Tuner 提供了四种调节器：`RandomSearch`、`Hyperband`、`BayesianOptimization` 和 `Sklearn`。在本教程中，您将使用 [Hyperband](https://arxiv.org/pdf/1603.06560.pdf) 调节器。

要实例化 Hyperband 调节器，必须指定超模型、要优化的 `objective` 和要训练的最大周期数 (`max_epochs`)。

```python
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
2021-08-13 23:32:45.391821: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.398545: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.399450: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.401464: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-08-13 23:32:45.401961: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.402870: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.403703: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.982879: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.983814: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.984692: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-13 23:32:45.985610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14648 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:05.0, compute capability: 7.0
```

Hyperband 调节算法使用自适应资源分配和早停法来快速收敛到高性能模型。该过程采用了体育竞技争冠模式的排除法。算法会将大量模型训练多个周期，并仅将性能最高的一半模型送入下一轮训练。Hyperband 通过计算 1 + log`factor`(`max_epochs`) 并将其向上舍入到最接近的整数来确定要训练的模型的数量。

创建回调以在验证损失达到特定值后提前停止训练。

```python
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
```

运行超参数搜索。除了上面的回调外，搜索方法的参数也与 `tf.keras.model.fit` 所用参数相同。

```python
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
Trial 30 Complete [00h 00m 28s]
val_accuracy: 0.8572499752044678

Best val_accuracy So Far: 0.8910833597183228
Total elapsed time: 00h 05m 57s
INFO:tensorflow:Oracle triggered exit

The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is 192 and the optimal learning rate for the optimizer
is 0.001.
```

## 训练模型

使用从搜索中获得的超参数找到训练模型的最佳周期数。

```python
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
Epoch 1/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.5083 - accuracy: 0.8213 - val_loss: 0.4035 - val_accuracy: 0.8567
Epoch 2/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3787 - accuracy: 0.8627 - val_loss: 0.3563 - val_accuracy: 0.8717
Epoch 3/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3383 - accuracy: 0.8765 - val_loss: 0.3508 - val_accuracy: 0.8726
Epoch 4/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3122 - accuracy: 0.8845 - val_loss: 0.3306 - val_accuracy: 0.8823
Epoch 5/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2937 - accuracy: 0.8919 - val_loss: 0.3411 - val_accuracy: 0.8777
Epoch 6/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2783 - accuracy: 0.8963 - val_loss: 0.3346 - val_accuracy: 0.8783
Epoch 7/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2659 - accuracy: 0.9022 - val_loss: 0.3126 - val_accuracy: 0.8873
Epoch 8/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2547 - accuracy: 0.9047 - val_loss: 0.3198 - val_accuracy: 0.8881
Epoch 9/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2427 - accuracy: 0.9080 - val_loss: 0.3335 - val_accuracy: 0.8798
Epoch 10/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2350 - accuracy: 0.9113 - val_loss: 0.3382 - val_accuracy: 0.8828
Epoch 11/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2272 - accuracy: 0.9145 - val_loss: 0.3195 - val_accuracy: 0.8879
Epoch 12/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2170 - accuracy: 0.9186 - val_loss: 0.3591 - val_accuracy: 0.8781
Epoch 13/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2102 - accuracy: 0.9209 - val_loss: 0.3257 - val_accuracy: 0.8857
Epoch 14/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2023 - accuracy: 0.9244 - val_loss: 0.3182 - val_accuracy: 0.8914
Epoch 15/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1976 - accuracy: 0.9264 - val_loss: 0.3444 - val_accuracy: 0.8870
Epoch 16/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1878 - accuracy: 0.9300 - val_loss: 0.3295 - val_accuracy: 0.8918
Epoch 17/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1847 - accuracy: 0.9301 - val_loss: 0.3330 - val_accuracy: 0.8895
Epoch 18/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1784 - accuracy: 0.9321 - val_loss: 0.3649 - val_accuracy: 0.8805
Epoch 19/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1732 - accuracy: 0.9347 - val_loss: 0.3525 - val_accuracy: 0.8887
Epoch 20/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1680 - accuracy: 0.9368 - val_loss: 0.3640 - val_accuracy: 0.8864
Epoch 21/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1630 - accuracy: 0.9385 - val_loss: 0.3413 - val_accuracy: 0.8915
Epoch 22/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1607 - accuracy: 0.9403 - val_loss: 0.3853 - val_accuracy: 0.8861
Epoch 23/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1544 - accuracy: 0.9429 - val_loss: 0.3879 - val_accuracy: 0.8791
Epoch 24/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1503 - accuracy: 0.9444 - val_loss: 0.3667 - val_accuracy: 0.8907
Epoch 25/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1460 - accuracy: 0.9453 - val_loss: 0.3672 - val_accuracy: 0.8932
Epoch 26/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1408 - accuracy: 0.9473 - val_loss: 0.3802 - val_accuracy: 0.8888
Epoch 27/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1390 - accuracy: 0.9469 - val_loss: 0.3746 - val_accuracy: 0.8868
Epoch 28/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1336 - accuracy: 0.9503 - val_loss: 0.4170 - val_accuracy: 0.8836
Epoch 29/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1315 - accuracy: 0.9513 - val_loss: 0.3973 - val_accuracy: 0.8913
Epoch 30/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1273 - accuracy: 0.9521 - val_loss: 0.3892 - val_accuracy: 0.8908
Epoch 31/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1247 - accuracy: 0.9534 - val_loss: 0.4113 - val_accuracy: 0.8883
Epoch 32/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1213 - accuracy: 0.9540 - val_loss: 0.4013 - val_accuracy: 0.8907
Epoch 33/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1178 - accuracy: 0.9555 - val_loss: 0.3917 - val_accuracy: 0.8928
Epoch 34/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1150 - accuracy: 0.9573 - val_loss: 0.4220 - val_accuracy: 0.8902
Epoch 35/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1138 - accuracy: 0.9570 - val_loss: 0.4507 - val_accuracy: 0.8827
Epoch 36/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1094 - accuracy: 0.9588 - val_loss: 0.4290 - val_accuracy: 0.8900
Epoch 37/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1071 - accuracy: 0.9604 - val_loss: 0.4409 - val_accuracy: 0.8915
Epoch 38/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1069 - accuracy: 0.9603 - val_loss: 0.4547 - val_accuracy: 0.8901
Epoch 39/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1026 - accuracy: 0.9619 - val_loss: 0.4780 - val_accuracy: 0.8862
Epoch 40/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1005 - accuracy: 0.9621 - val_loss: 0.4572 - val_accuracy: 0.8892
Epoch 41/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0987 - accuracy: 0.9635 - val_loss: 0.4462 - val_accuracy: 0.8910
Epoch 42/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0972 - accuracy: 0.9629 - val_loss: 0.4482 - val_accuracy: 0.8881
Epoch 43/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0918 - accuracy: 0.9655 - val_loss: 0.4698 - val_accuracy: 0.8861
Epoch 44/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0926 - accuracy: 0.9658 - val_loss: 0.4755 - val_accuracy: 0.8896
Epoch 45/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0911 - accuracy: 0.9656 - val_loss: 0.4781 - val_accuracy: 0.8877
Epoch 46/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0868 - accuracy: 0.9667 - val_loss: 0.5025 - val_accuracy: 0.8812
Epoch 47/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0888 - accuracy: 0.9668 - val_loss: 0.5001 - val_accuracy: 0.8893
Epoch 48/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0842 - accuracy: 0.9688 - val_loss: 0.5031 - val_accuracy: 0.8919
Epoch 49/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0873 - accuracy: 0.9675 - val_loss: 0.5087 - val_accuracy: 0.8896
Epoch 50/50
1500/1500 [==============================] - 3s 2ms/step - loss: 0.0828 - accuracy: 0.9691 - val_loss: 0.5169 - val_accuracy: 0.8861
Best epoch: 25
```

重新实例化超模型并使用上面的最佳周期数对其进行训练。

```python
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
Epoch 1/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.5081 - accuracy: 0.8212 - val_loss: 0.4262 - val_accuracy: 0.8482
Epoch 2/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3800 - accuracy: 0.8621 - val_loss: 0.3819 - val_accuracy: 0.8620
Epoch 3/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3393 - accuracy: 0.8757 - val_loss: 0.3642 - val_accuracy: 0.8677
Epoch 4/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3122 - accuracy: 0.8856 - val_loss: 0.3330 - val_accuracy: 0.8806
Epoch 5/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2922 - accuracy: 0.8929 - val_loss: 0.3187 - val_accuracy: 0.8857
Epoch 6/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2789 - accuracy: 0.8970 - val_loss: 0.3211 - val_accuracy: 0.8846
Epoch 7/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2632 - accuracy: 0.9026 - val_loss: 0.3246 - val_accuracy: 0.8867
Epoch 8/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2523 - accuracy: 0.9055 - val_loss: 0.3238 - val_accuracy: 0.8844
Epoch 9/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2410 - accuracy: 0.9097 - val_loss: 0.3218 - val_accuracy: 0.8889
Epoch 10/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2314 - accuracy: 0.9138 - val_loss: 0.3208 - val_accuracy: 0.8882
Epoch 11/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2245 - accuracy: 0.9162 - val_loss: 0.3310 - val_accuracy: 0.8842
Epoch 12/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2163 - accuracy: 0.9186 - val_loss: 0.3241 - val_accuracy: 0.8900
Epoch 13/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2091 - accuracy: 0.9217 - val_loss: 0.3155 - val_accuracy: 0.8937
Epoch 14/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.2012 - accuracy: 0.9245 - val_loss: 0.3403 - val_accuracy: 0.8843
Epoch 15/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1958 - accuracy: 0.9265 - val_loss: 0.3183 - val_accuracy: 0.8937
Epoch 16/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1876 - accuracy: 0.9279 - val_loss: 0.3530 - val_accuracy: 0.8829
Epoch 17/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1847 - accuracy: 0.9312 - val_loss: 0.3380 - val_accuracy: 0.8898
Epoch 18/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1761 - accuracy: 0.9328 - val_loss: 0.3430 - val_accuracy: 0.8936
Epoch 19/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1720 - accuracy: 0.9356 - val_loss: 0.3345 - val_accuracy: 0.8899
Epoch 20/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1670 - accuracy: 0.9373 - val_loss: 0.3519 - val_accuracy: 0.8866
Epoch 21/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1627 - accuracy: 0.9387 - val_loss: 0.3839 - val_accuracy: 0.8836
Epoch 22/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1568 - accuracy: 0.9408 - val_loss: 0.3482 - val_accuracy: 0.8919
Epoch 23/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1540 - accuracy: 0.9425 - val_loss: 0.3765 - val_accuracy: 0.8915
Epoch 24/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1480 - accuracy: 0.9441 - val_loss: 0.3674 - val_accuracy: 0.8888
Epoch 25/25
1500/1500 [==============================] - 3s 2ms/step - loss: 0.1430 - accuracy: 0.9466 - val_loss: 0.3506 - val_accuracy: 0.8925
<keras.callbacks.History at 0x7f3bec2c2850>
```

要完成本教程，请在测试数据上评估超模型。

```python
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)
313/313 [==============================] - 1s 2ms/step - loss: 0.3943 - accuracy: 0.8890
[test loss, test accuracy]: [0.3942869007587433, 0.8889999985694885]
```

`my_dir/intro_to_kt` 目录中包含了在超参数搜索期间每次试验（模型配置）运行的详细日志和检查点。如果重新运行超参数搜索，Keras Tuner 将使用这些日志中记录的现有状态来继续搜索。要停用此行为，请在实例化调节器时传递一个附加的 `overwrite = True` 参数。

## 总结

在本教程中，您学习了如何使用 Keras Tuner 调节模型的超参数。要详细了解 Keras Tuner，请查看以下其他资源：

- [TensorFlow 博客上的 Keras Tuner](https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html)
- [Keras Tuner 网站](https://keras-team.github.io/keras-tuner/)

另请查看 TensorBoard 中的 [HParams Dashboard](https://tensorflow.google.cn/tensorboard/hyperparameter_tuning_with_hparams)，以交互方式调节模型超参数。