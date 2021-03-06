# 初学者的 TensorFlow 2.0 教程

这是一个 [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) 笔记本文件。 Python程序可以直接在浏览器中运行，这是学习 Tensorflow 的绝佳方式。想要学习该教程，请点击此页面顶部的按钮，在Google Colab中运行笔记本。

1. 在 Colab中, 连接到Python运行环境： 在菜单条的右上方, 选择 *CONNECT*。
2. 运行所有的代码块: 选择 *Runtime* > *Run all*。

下载并安装 TensorFlow 2.0 测试版包。将 TensorFlow 载入你的程序：

```python
# 安装 TensorFlow

import tensorflow as tf
```

载入并准备好 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)。将样本从整数转换为浮点数：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

将模型的各层堆叠起来，以搭建 `tf.keras.Sequential` 模型。为训练选择优化器和损失函数：

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

训练并验证模型：

```python
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
Epoch 1/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2962 - accuracy: 0.9155
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1420 - accuracy: 0.9581
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1064 - accuracy: 0.9672
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0885 - accuracy: 0.9730
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0749 - accuracy: 0.9765
313/313 - 0s - loss: 0.0748 - accuracy: 0.9778
[0.07484959065914154, 0.9778000116348267]
```

现在，这个照片分类器的准确度已经达到 98%。想要了解更多，请阅读 [TensorFlow 教程](https://tensorflow.google.cn/tutorials/)。

