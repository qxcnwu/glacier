from Efficinet import cnn_model
from made_record import tfrecords_reader_dataset
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras import callbacks
import os

train_record="test.tfrecord"
test_record="test.tfrecord"
batch_size = 2
epoch=1000

# 预选退火学习率
class CosineAnnealing(callbacks.Callback):
    """Cosine annealing according to DECOUPLED WEIGHT DECAY REGULARIZATION.

    # Arguments
        eta_max: float, eta_max in eq(5).
        eta_min: float, eta_min in eq(5).
        total_iteration: int, Ti in eq(5).
        iteration: int, T_cur in eq(5).
        verbose: 0 or 1.
    """

    def __init__(self, eta_max=1, eta_min=0., total_iteration=0, iteration=3, verbose=1, **kwargs):
        super(CosineAnnealing, self).__init__()

        global lr_list

        lr_list = []
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.total_iteration = total_iteration
        self.iteration = iteration

    def on_train_begin(self, logs=None):
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_train_end(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_batch_end(self, epoch, logs=None):
        self.iteration += 1
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * 0.5 * (
                1 + np.cos(np.pi * self.iteration / self.total_iteration))
        new_lr=K.clip(self.lr * eta_t,0.00001,1)
        K.set_value(self.model.optimizer.lr, new_lr)
# 交叉熵损失函数
def weighted_loss(y_true,y_pred):
    w_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.squeeze(y_true))
    return K.mean(w_loss)
# 正确率IOU交并比计算
def accu(y_true,y_pred):
    y_pred=K.argmax(y_pred,axis=1)
    y_true=K.cast(y_true,dtype=tf.int64)
    true=K.equal(y_true,y_pred)
    acc=K.sum(K.cast(true,dtype=tf.int64))/batch_size
    return acc
# 损失函数
def my_loss(y_true,y_pred):
    return weighted_loss(y_true,y_pred)


# 网络主体
model=cnn_model()
# model.load_weights('model.h5')
model.compile(optimizer=keras.optimizers.Adam(),loss=my_loss, metrics=[accu])
model.summary()

# 训练测试数据读取
valid_data = tfrecords_reader_dataset(test_record, batch_size=batch_size)
train_data =tfrecords_reader_dataset(train_record, batch_size=batch_size)

print(valid_data,train_data)

epochs = [3*i for i in range(1,30)]
# 训练过程
for Ti in epochs:
    print(Ti)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='val_loss', verbose=1,save_best_only=False, save_weights_only=False, period=1)
    reduce_lr = CosineAnnealing(eta_max=1, eta_min=0.00001, total_iteration=Ti * (int(25876*0.8) // batch_size), iteration=3, verbose=1)
    model.fit_generator(train_data, steps_per_epoch=int(25876 * 0.8 / batch_size), epochs=Ti,callbacks=[model_checkpoint, reduce_lr], shuffle=True,validation_data=valid_data, validation_steps=int(25876 * 0.2 / batch_size))

