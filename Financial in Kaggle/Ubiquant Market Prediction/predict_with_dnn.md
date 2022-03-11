### simple deep neural network

1. use LeakyRuLU activation
2. use BatchNormalization
3. use Dropout
4. use kernel_initializer with 'he_normal'
5. use ExponentialDecay scheduling
6. use ModelCheckpoint

### tips:

1. 适用parquet加载数据，train数据18.55G，内存总共只有13G可用
2. 使用所有301个features，可以使用LightGBM 来使用部分features
3. 使用sklearn的StandardScaler来标准化数据
4. DNN



```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib as plt

import tensorflow as tf

import ubiquant
print(tf.__version__)
2.6.2
```

In [2]:

```python
"""Load dataset
总数据量为18.55G, 先使用部分数据进行测试

全量训练数据加载可以使用ubiquant-parquet加载数据

"""
path = "../input/ubiquant-market-prediction/train.csv"
# train = pd.read_csv(path, nrows = 100000)
# load train.parquet instead
path_train_parquet = "../input/ubiquant-parquet/train_low_mem.parquet"
train = pd.read_parquet(path_train_parquet)
train.describe()
```

Out[2]:

|       | time_id      | investment_id | target        | f_0           | f_1           | f_2           | f_3           | f_4           | f_5           | f_6           | ...  | f_290         | f_291         | f_292         | f_293         | f_294         | f_295         | f_296         | f_297         | f_298         | f_299         |
| :---- | :----------- | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :--- | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ | :------------ |
| count | 3.141410e+06 | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | ...  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  | 3.141410e+06  |
| mean  | 6.654862e+02 | 1.885265e+03  | -2.109159e-02 | 1.637057e-03  | -1.257678e-02 | 8.514749e-03  | -2.003703e-02 | -5.416438e-03 | -3.034008e-03 | 1.934330e-03  | ...  | 1.256709e-02  | 9.002053e-04  | -6.509154e-03 | 2.851608e-03  | -4.730820e-03 | -2.773806e-02 | -1.517383e-03 | -7.057928e-03 | -4.216896e-03 | -4.319488e-03 |
| std   | 3.560434e+02 | 1.083616e+03  | 9.176629e-01  | 1.079721e+00  | 1.030980e+00  | 1.030281e+00  | 9.602916e-01  | 9.895188e-01  | 1.104578e+00  | 1.067755e+00  | ...  | 1.084496e+00  | 1.114387e+00  | 1.070176e+00  | 1.101158e+00  | 1.144516e+00  | 9.592550e-01  | 1.140459e+00  | 1.108097e+00  | 1.051252e+00  | 1.008220e+00  |
| min   | 0.000000e+00 | 0.000000e+00  | -9.419646e+00 | -1.765789e+01 | -6.579473e+00 | -8.644268e+00 | -1.800427e+01 | -4.000015e+00 | -8.833704e+00 | -6.384251e+00 | ...  | -9.271487e+00 | -6.359966e+00 | -8.183732e+00 | -2.399478e+01 | -1.497270e+01 | -9.348986e+00 | -7.366648e+00 | -7.579406e+00 | -6.707284e+00 | -1.028264e+01 |
| 25%   | 3.530000e+02 | 9.520000e+02  | -5.004572e-01 | -4.071441e-01 | -6.813878e-01 | -6.530838e-01 | -4.496046e-01 | -3.532303e-01 | -7.070855e-01 | -7.911198e-01 | ...  | -6.037326e-01 | -8.355157e-01 | -6.981400e-01 | -1.377131e-01 | -9.397985e-01 | -5.147926e-01 | -9.573807e-01 | -7.239380e-01 | -8.165390e-01 | -5.080143e-01 |
| 50%   | 7.040000e+02 | 1.882000e+03  | -9.717009e-02 | 2.431158e-01  | -4.867587e-02 | 4.350941e-02  | -2.639937e-01 | -1.908876e-01 | -2.144796e-02 | 1.437945e-02  | ...  | 6.086323e-02  | -2.023181e-01 | -1.910102e-01 | 2.164071e-01  | 2.087202e-01  | -2.907780e-01 | 6.955573e-03  | -1.650222e-01 | 2.300689e-02  | -2.824031e-01 |
| 75%   | 9.750000e+02 | 2.830000e+03  | 3.572908e-01  | 6.649507e-01  | 6.086557e-01  | 6.587324e-01  | 7.763371e-02  | 2.784694e-02  | 6.835684e-01  | 8.008306e-01  | ...  | 6.882384e-01  | 9.035954e-01  | 4.841421e-01  | 5.088849e-01  | 8.616266e-01  | 1.275946e-01  | 9.503851e-01  | 6.098197e-01  | 7.984827e-01  | 1.411301e-01  |
| max   | 1.219000e+03 | 3.773000e+03  | 1.203861e+01  | 7.845261e+00  | 8.707207e+00  | 8.009340e+00  | 4.706333e+01  | 7.662866e+01  | 7.646200e+00  | 6.778142e+00  | ...  | 9.298274e+00  | 9.725060e+00  | 2.231185e+01  | 6.587691e+00  | 6.978151e+00  | 6.140367e+01  | 7.679950e+00  | 1.241804e+01  | 7.003982e+00  | 4.337021e+01  |

8 rows × 303 columns

In [3]:

```python
"""use all features
你也可以使用部分参数，可以参考Lgbm

"""
train.columns
```

Out[3]:

```
Index(['row_id', 'time_id', 'investment_id', 'target', 'f_0', 'f_1', 'f_2',
       'f_3', 'f_4', 'f_5',
       ...
       'f_290', 'f_291', 'f_292', 'f_293', 'f_294', 'f_295', 'f_296', 'f_297',
       'f_298', 'f_299'],
      dtype='object', length=304)
```

In [4]:

```
# use features: f_0 ~ f_299
f_col = train.drop(['row_id','time_id','investment_id','target'],axis=1).columns
f_col
```

Out[4]:

```
Index(['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9',
       ...
       'f_290', 'f_291', 'f_292', 'f_293', 'f_294', 'f_295', 'f_296', 'f_297',
       'f_298', 'f_299'],
      dtype='object', length=300)
```

In [5]:

```
# 使用sklearn的StandardScaler来标准化数据
scaler = StandardScaler()
# type(train['investment_id'])
scaler.fit(pd.DataFrame(train['investment_id']))
```

Out[5]:

```
StandardScaler()
```

In [6]:

```
# make dataset function
def make_dataset(df):
    inv_df = df['investment_id']
    f_df = df[f_col]
    scaled_investment_id = scaler.transform(pd.DataFrame(inv_df))
    df['investment_id'] = scaled_investment_id
    data_x = pd.concat([df['investment_id'], f_df], axis=1)
    return data_x
```

In [7]:

```
# change the data type cause the memory limited which is too small to use raw data.
train = train.astype('float16')
train_x = make_dataset(train)
train_x
```

Out[7]:

|         | investment_id | f_0       | f_1       | f_2       | f_3       | f_4       | f_5       | f_6       | f_7       | f_8      | ...  | f_290     | f_291     | f_292     | f_293     | f_294     | f_295     | f_296     | f_297     | f_298     | f_299     |
| :------ | :------------ | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :------- | :--- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- | :-------- |
| 0       | -1.738281     | 0.932617  | 0.113708  | -0.402100 | 0.378418  | -0.203979 | -0.413574 | 0.965820  | 1.230469  | 0.114807 | ...  | 0.365967  | -1.095703 | 0.200073  | 0.819336  | 0.941406  | -0.086792 | -1.086914 | -1.044922 | -0.287598 | 0.321533  |
| 1       | -1.737305     | 0.811035  | -0.514160 | 0.742188  | -0.616699 | -0.194214 | 1.771484  | 1.427734  | 1.133789  | 0.114807 | ...  | -0.154175 | 0.912598  | -0.734375 | 0.819336  | 0.941406  | -0.387695 | -1.086914 | -0.929688 | -0.974121 | -0.343506 |
| 2       | -1.734375     | 0.394043  | 0.615723  | 0.567871  | -0.607910 | 0.068909  | -1.083008 | 0.979492  | -1.125977 | 0.114807 | ...  | -0.138062 | 0.912598  | -0.551758 | -1.220703 | -1.060547 | -0.219116 | -1.086914 | -0.612305 | -0.113953 | 0.243652  |
| 3       | -1.733398     | -2.343750 | -0.011871 | 1.875000  | -0.606445 | -0.586914 | -0.815918 | 0.778320  | 0.299072  | 0.114807 | ...  | 0.382080  | 0.912598  | -0.266357 | -1.220703 | 0.941406  | -0.608887 | 0.104919  | -0.783203 | 1.151367  | -0.773438 |
| 4       | -1.732422     | 0.842285  | -0.262939 | 2.330078  | -0.583496 | -0.618164 | -0.742676 | -0.946777 | 1.230469  | 0.114807 | ...  | -0.170410 | 0.912598  | -0.741211 | -1.220703 | 0.941406  | -0.588379 | 0.104919  | 0.753418  | 1.345703  | -0.737793 |
| ...     | ...           | ...       | ...       | ...       | ...       | ...       | ...       | ...       | ...       | ...      | ...  | ...       | ...       | ...       | ...       | ...       | ...       | ...       | ...       | ...       | ...       |
| 3141405 | 1.737305      | 0.093506  | -0.720215 | -0.345459 | -0.438721 | -0.166992 | -0.437256 | 1.475586  | 1.284180  | 0.056427 | ...  | -0.285889 | -1.232422 | -0.660645 | 0.875488  | 0.421631  | -0.427979 | -0.075562 | -0.533203 | -0.193726 | -0.581543 |
| 3141406 | 1.737305      | -1.344727 | -0.199951 | -0.107727 | -0.454590 | -0.221924 | -0.141113 | -1.498047 | 1.374023  | 0.056427 | ...  | 0.184570  | -1.232422 | -0.670410 | 0.875488  | 0.421631  | -0.729980 | -1.514648 | 0.013145  | -0.890137 | -0.589844 |
| 3141407 | 1.739258      | 0.979492  | -1.110352 | 1.006836  | -0.467285 | -0.159546 | 1.355469  | 0.150757  | -0.088928 | 0.056427 | ...  | -0.756348 | -1.232422 | 0.820801  | -1.142578 | 0.421631  | -0.363281 | 1.363281  | -0.079102 | -1.580078 | -0.297607 |
| 3141408 | 1.741211      | -2.564453 | 0.320312  | 0.076599  | 1.379883  | -0.155396 | -0.688965 | 0.381104  | -1.325195 | 0.056427 | ...  | -0.756348 | -1.232422 | 0.133057  | -1.142578 | 0.421631  | -0.375244 | -1.514648 | -0.973633 | 0.608887  | -0.372070 |
| 3141409 | 1.741211      | -0.089539 | 0.190186  | -0.548340 | 0.151245  | 0.079773  | 0.447998  | 1.014648  | -1.325195 | 0.056427 | ...  | -0.317139 | 0.811523  | 3.271484  | 0.875488  | 0.421631  | -0.170654 | 1.363281  | -0.563477 | 0.669434  | 0.456299  |

3141410 rows × 301 columns

In [8]:

```
# Target variable
train_y = pd.DataFrame(train['target'])
train_y
```

Out[8]:

|         | target    |
| :------ | :-------- |
| 0       | -0.300781 |
| 1       | -0.231079 |
| 2       | 0.568848  |
| 3       | -1.064453 |
| 4       | -0.531738 |
| ...     | ...       |
| 3141405 | 0.033600  |
| 3141406 | -0.223267 |
| 3141407 | -0.559570 |
| 3141408 | 0.009598  |
| 3141409 | 1.211914  |

3141410 rows × 1 columns

In [9]:

```
# this code cell is for test
train_x.shape
train_x.shape[1]
```

Out[9]:

```
301
```

In [10]:

```
"""Model
simple deep neural network
1. use LeakyRuLU activation
2. use BatchNormalization
3. use Dropout
4. use kernel_initializer with 'he_normal'
5. use ExponentialDecay scheduling
6. use ModelCheckpoint

"""
def make_model():
    # 设置输入维度 301
    inputs_ = tf.keras.Input(shape = [train_x.shape[1]])
    # 初始化权重
    x = tf.keras.layers.Dense(64, kernel_initializer = 'he_normal')(inputs_)
    # 标准化
    # 训练过程中，面对不同的数据batch时，神经网络每一层的各自的输入数据分布应自始至终保持一致，
    # 未能保持一致的现象叫做Internal Covaraite Shift，会影响训练过程。
    batch = tf.keras.layers.BatchNormalization()(x)
    # 设置激活函数，适用LeakyRuLU
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    
    x = tf.keras.layers.Dense(128, kernel_initializer = 'he_normal')(leaky)
    batch = tf.keras.layers.BatchNormalization()(x)
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    
    x = tf.keras.layers.Dense(256, kernel_initializer = 'he_normal')(leaky)
    batch = tf.keras.layers.BatchNormalization()(x)
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    
    x = tf.keras.layers.Dense(512, kernel_initializer = 'he_normal')(leaky)
    batch = tf.keras.layers.BatchNormalization()(x)
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    
    x = tf.keras.layers.Dense(256, kernel_initializer = 'he_normal')(leaky)
    batch = tf.keras.layers.BatchNormalization()(x)
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    drop = tf.keras.layers.Dropout(0.4)(leaky)
    
    x = tf.keras.layers.Dense(128, kernel_initializer = 'he_normal')(drop)
    batch = tf.keras.layers.BatchNormalization()(x)
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    
    x = tf.keras.layers.Dense(8, kernel_initializer = 'he_normal')(leaky)
    batch = tf.keras.layers.BatchNormalization()(x)
    leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    drop = tf.keras.layers.Dropout(0.4)(leaky)
    
    # 最终输出
    outputs_ = tf.keras.layers.Dense(1)(drop)
    
    model = tf.keras.Model(inputs = inputs_, outputs = outputs_)
    
    rmse = tf.keras.metrics.RootMeanSquaredError()
    
    learning_sch = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.003,
    decay_steps = 9700,
    decay_rate = 0.98)
    adam = tf.keras.optimizers.Adam(learning_rate = learning_sch)
    
    model.compile(loss = 'mse', metrics = rmse, optimizer = adam)
    return model
    
make_model().summary()
```

```
2022-03-11 07:45:42.260956: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:42.352706: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:42.353360: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:42.355608: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-11 07:45:42.355896: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:42.356552: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:42.357161: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:44.100401: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:44.101429: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:44.102194: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-11 07:45:44.103548: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15403 MB memory:  -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 301)]             0         
_________________________________________________________________
dense (Dense)                (None, 64)                19328     
_________________________________________________________________
batch_normalization (BatchNo (None, 64)                256       
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               8320      
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               33024     
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 512)               131584    
_________________________________________________________________
batch_normalization_3 (Batch (None, 512)               2048      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 512)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               131328    
_________________________________________________________________
batch_normalization_4 (Batch (None, 256)               1024      
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 128)               32896     
_________________________________________________________________
batch_normalization_5 (Batch (None, 128)               512       
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 128)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 8)                 1032      
_________________________________________________________________
batch_normalization_6 (Batch (None, 8)                 32        
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 8)                 0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 8)                 0         
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 9         
=================================================================
Total params: 362,929
Trainable params: 360,225
Non-trainable params: 2,704
_________________________________________________________________
```

In [11]:

```
# model graphic
tf.keras.utils.plot_model(make_model(),show_shapes=True,expand_nested=True)
```

Out[11]:

![img](https://www.kaggleusercontent.com/kf/89817791/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..pXTmFsej87_-LTZwxSx5Nw.oDzfPWRXSVNxlYn_rl9ClA2_3SNpK2rywcXQqPzda1hsY2kcEzVyOMjYk7HGcX2uUpPma86a380oqiyiivf2SGhgrC5y9EMLKko23sz08tndGueIhHwGCD0Nmgxe7aP1JKaGXE5XeBv2nJW-dBtQkJVO_Vk331wy-T30I74BbjRqHWMoXhy3GTNTvj41QuufXn0lf6YQ3hLjwHr-mgsp4smH2PTIYWfMEb6OZYNo-wYCms18eGFqHa5_hkyB_lxx1FQViGIxJ03XgviTMwZZVjPYYb_tDplfYMUsQ8ExGWWM_vjG13jjxrlkvoejhMqxfkF9-mLPX4IRgBJcgDSzMiR2C03BxtDQJvYPK4GFcIoxGxPX3GpgiGCwyxr0YouJazXk6eDfsSwDXRj1_J1F46an5S-eUdV2cxBep6Wcy2hs-4GnQ5lH2LW3rD_byzppbG4VyLGDQuDzmwVR5kTUEYds5_SSoeqxy7QyCc3b9Jhj00zBYJIXHIL3fY-Iswj8CSzFyfa2XFTf86-d2OnU1aJluSsVyHtGSkBOE7AGjMhvQ57jFn0tmmiRb5MRHg7aO4IQtmxC_0XwFj_eRESuTrj2RLibcBQVtQsC6spQBDtj6vXczHJ_ik3b2YQLRjA4qASWcjLExAAr68epSJGviSgPuLam5l92ar771G7Hb4Y._58q0lI__q0irWZCcy4Rjw/__results___files/__results___12_0.png)

In [12]:

```
# KFold strategy
kfold_generator = KFold(n_splits =5, shuffle=True, random_state = 2022)
kfold_generator
```

Out[12]:

```
KFold(n_splits=5, random_state=2022, shuffle=True)
```

In [13]:

```
# model fitting
callbacks = tf.keras.callbacks.ModelCheckpoint('dnn_model.h5', save_best_only = True)

# 命名的转换
df_x = train_x
df_y = train_y
for train_index, val_index in kfold_generator.split(df_x, df_y):
    # Split training dataset.
    train_x, train_y = df_x.iloc[train_index], df_y.iloc[train_index]
    # Split validation dataset.
    val_x, val_y = df_x.iloc[val_index], df_y.iloc[val_index]
    # Make tensor dataset.
    tf_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(2022).batch(1024, drop_remainder=True).prefetch(1)
    tf_val = tf.data.Dataset.from_tensor_slices((val_x, val_y)).shuffle(2022).batch(1024, drop_remainder=True).prefetch(1)
    # Load model
    model = make_model()
    # Model fitting
    
    ## I used 5 epochs for fast save.
    ## Change the epochs into more numbers.
    model.fit(tf_train, callbacks = callbacks, epochs = 5, #### change the epochs into more numbers.
             validation_data = (tf_val), shuffle=True)
    # Delete tensor dataset and model for avoiding memory exploring.
    del tf_train
    del tf_val
    del model
    
2022-03-11 07:45:56.992035: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1512903056 exceeds 10% of free system memory.
2022-03-11 07:45:58.661197: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1512903056 exceeds 10% of free system memory.
2022-03-11 07:46:00.900832: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1512903056 exceeds 10% of free system memory.
2022-03-11 07:46:01.850045: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1512903056 exceeds 10% of free system memory.
Epoch 1/5
2022-03-11 07:46:04.185300: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2454/2454 [==============================] - 20s 7ms/step - loss: 0.8410 - root_mean_squared_error: 0.9171 - val_loss: 0.8462 - val_root_mean_squared_error: 0.9199
2022-03-11 07:46:22.600089: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 1512903056 exceeds 10% of free system memory.
Epoch 2/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8314 - root_mean_squared_error: 0.9118 - val_loss: 0.8419 - val_root_mean_squared_error: 0.9175
Epoch 3/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8301 - root_mean_squared_error: 0.9111 - val_loss: 0.8441 - val_root_mean_squared_error: 0.9188
Epoch 4/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8293 - root_mean_squared_error: 0.9107 - val_loss: 0.8433 - val_root_mean_squared_error: 0.9183
Epoch 5/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8284 - root_mean_squared_error: 0.9102 - val_loss: 0.8413 - val_root_mean_squared_error: 0.9172
Epoch 1/5
2454/2454 [==============================] - 19s 7ms/step - loss: 0.8476 - root_mean_squared_error: 0.9207 - val_loss: 0.8361 - val_root_mean_squared_error: 0.9144
Epoch 2/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8327 - root_mean_squared_error: 0.9125 - val_loss: 0.8385 - val_root_mean_squared_error: 0.9157
Epoch 3/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8316 - root_mean_squared_error: 0.9119 - val_loss: 0.8369 - val_root_mean_squared_error: 0.9148
Epoch 4/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8306 - root_mean_squared_error: 0.9114 - val_loss: 0.8342 - val_root_mean_squared_error: 0.9134
Epoch 5/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8294 - root_mean_squared_error: 0.9107 - val_loss: 0.8340 - val_root_mean_squared_error: 0.9132
Epoch 1/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8450 - root_mean_squared_error: 0.9192 - val_loss: 0.8311 - val_root_mean_squared_error: 0.9117
Epoch 2/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8334 - root_mean_squared_error: 0.9129 - val_loss: 0.8310 - val_root_mean_squared_error: 0.9116
Epoch 3/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8322 - root_mean_squared_error: 0.9123 - val_loss: 0.8310 - val_root_mean_squared_error: 0.9116
Epoch 4/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8312 - root_mean_squared_error: 0.9117 - val_loss: 0.8301 - val_root_mean_squared_error: 0.9111
Epoch 5/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8303 - root_mean_squared_error: 0.9112 - val_loss: 0.8299 - val_root_mean_squared_error: 0.9110
Epoch 1/5
2454/2454 [==============================] - 20s 7ms/step - loss: 0.8531 - root_mean_squared_error: 0.9236 - val_loss: 0.8371 - val_root_mean_squared_error: 0.9149
Epoch 2/5
2454/2454 [==============================] - 16s 7ms/step - loss: 0.8323 - root_mean_squared_error: 0.9123 - val_loss: 0.8392 - val_root_mean_squared_error: 0.9161
Epoch 3/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8314 - root_mean_squared_error: 0.9118 - val_loss: 0.8361 - val_root_mean_squared_error: 0.9144
Epoch 4/5
2454/2454 [==============================] - 18s 7ms/step - loss: 0.8304 - root_mean_squared_error: 0.9113 - val_loss: 0.8353 - val_root_mean_squared_error: 0.9139
Epoch 5/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8295 - root_mean_squared_error: 0.9108 - val_loss: 0.8346 - val_root_mean_squared_error: 0.9136
Epoch 1/5
2454/2454 [==============================] - 20s 7ms/step - loss: 0.8411 - root_mean_squared_error: 0.9171 - val_loss: 0.8428 - val_root_mean_squared_error: 0.9180
Epoch 2/5
2454/2454 [==============================] - 18s 8ms/step - loss: 0.8323 - root_mean_squared_error: 0.9123 - val_loss: 0.8408 - val_root_mean_squared_error: 0.9170
Epoch 3/5
2454/2454 [==============================] - 16s 7ms/step - loss: 0.8311 - root_mean_squared_error: 0.9117 - val_loss: 0.8413 - val_root_mean_squared_error: 0.9172
Epoch 4/5
2454/2454 [==============================] - 19s 8ms/step - loss: 0.8299 - root_mean_squared_error: 0.9110 - val_loss: 0.8404 - val_root_mean_squared_error: 0.9167
Epoch 5/5
2454/2454 [==============================] - 17s 7ms/step - loss: 0.8290 - root_mean_squared_error: 0.9105 - val_loss: 0.8379 - val_root_mean_squared_error: 0.9154
```

In [14]:

```python
# submit
best_model = tf.keras.models.load_model('dnn_model.h5')
# initialize the environment
env = ubiquant.make_env()
 # an iterator which loops over the test set and sample submission
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df = make_dataset(test_df)
    sample_prediction_df['target'] = best_model.predict(test_df)  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
This version of the API is not optimized and should not be used to estimate the runtime of your code on the hidden test set.
```