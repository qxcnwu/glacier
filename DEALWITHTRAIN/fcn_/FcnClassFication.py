import os
# from Efficiunet import UEfficientNet
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow.keras.layers as k
import tensorflow.keras as keras
import tensorflow as tf
import efficientnet.tfkeras as efn


# 卷积Block块
def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = k.SeparableConv2D(filters, size, strides=strides, padding=padding)(x)
    x = k.BatchNormalization()(x)
    if activation == True:
        x = k.LeakyReLU(alpha=0.1)(x)
    return x
# 残差结构
def residual_block(blockInput, num_filters=16):
    x = k.LeakyReLU(alpha=0.1)(blockInput)
    x = k.BatchNormalization()(x)
    blockInput = k.BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = k.Add()([x, blockInput])
    return x
# Efficinet + Unet++
def UEfficientNet(input_shape=(None,None,3),dropout_rate=0.0):
    # out_channel=[3, 64, 48, 80, 224, 640]
    start_neurons=8
    backbone=efn.EfficientNetB6(input_shape=(256,256, 3),weights=None,include_top=False)
    input=backbone.input

    conv4 = backbone.layers[655].output
    conv4 = k.LeakyReLU(alpha=0.1)(conv4)
    pool4 = k.MaxPooling2D((2, 2))(conv4)
    pool4 = k.Dropout(dropout_rate)(pool4)

    convm = k.SeparableConv2D(start_neurons * 32, (3, 3), activation=None, padding="same", name='conv_middle')(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = k.LeakyReLU(alpha=0.1)(convm)

    deconv4=k.UpSampling2D(interpolation='bilinear')(convm)
    deconv4_up1 = k.UpSampling2D(interpolation='bilinear')(deconv4)
    deconv4_up2 = k.UpSampling2D(interpolation='bilinear')(deconv4_up1)
    deconv4_up3 = k.UpSampling2D(interpolation='bilinear')(deconv4_up2)
    uconv4 = k.concatenate([deconv4, conv4])
    uconv4 = k.Dropout(dropout_rate)(uconv4)

    uconv4 = k.SeparableConv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = k.LeakyReLU(alpha=0.1)(uconv4)  # conv1_2

    deconv3 = k.UpSampling2D(interpolation='bilinear')(uconv4)
    deconv3_up1 = k.UpSampling2D(interpolation='bilinear')(deconv3)
    deconv3_up2 = k.UpSampling2D(interpolation='bilinear')(deconv3_up1)
    conv3 = backbone.layers[432].output
    uconv3 = k.concatenate([deconv3, deconv4_up1, conv3])
    uconv3 = k.Dropout(dropout_rate)(uconv3)

    uconv3 = k.SeparableConv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = k.LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = k.UpSampling2D(interpolation='bilinear')(uconv3)
    deconv2_up1 = k.UpSampling2D(interpolation='bilinear')(deconv2)
    conv2 = backbone.layers[196].output
    uconv2 = k.concatenate([deconv2, deconv3_up1, deconv4_up2, conv2])

    uconv2 = k.Dropout(0.1)(uconv2)
    uconv2 = k.SeparableConv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = k.LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = k.UpSampling2D(interpolation='bilinear')(uconv2)
    conv1 = backbone.layers[108].output
    uconv1 = k.concatenate([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1])

    uconv1 = k.Dropout(0.1)(uconv1)
    uconv1 = k.SeparableConv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = k.LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = k.UpSampling2D((4,4),interpolation='bilinear')(uconv1)
    uconv0 = k.Dropout(0.1)(uconv0)
    uconv0 = k.SeparableConv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 2)
    uconv0 = k.LeakyReLU(alpha=0.1)(uconv0)
    uconv0 = k.Dropout(dropout_rate / 2)(uconv0)
    output_layer = k.Conv2D(2, (1, 1), padding="same", activation=None)(uconv0)

    model = keras.models.Model(inputs=input, outputs=output_layer)
    return model

class FCNDEAL:
    def __init__(self,data_array:np.array,npy_path:os.path,visio_dir:os.path,save_dir:os.path,model_path:os.path):
        """
        语义分割类
        :param data_array: 文件矩阵
        :param npy_path: npy文件路径
        :param save_dir: jpg文件保存路径
        :param visio_dir: visio文件保存路径
        :param model_path: FCN模型路径
        """
        print("FCN DOING!!...")
        self.data_array=data_array
        self.npy_dir=npy_path
        self.save_dir=save_dir
        self.visio_dir=visio_dir
        self.model = UEfficientNet()
        self.model.load_weights(model_path)
        self.H,self.W,_=self.data_array.shape

    def fcn_classcification(self,npy_path):
        data=np.load(npy_path)
        data=np.expand_dims(data,0)
        prediction = self.model.predict(data)
        ans = np.argmax(prediction, axis=-1)
        return np.reshape(ans, (256, 256))

    def save_ans(self,ans,jpg_path,visio_path):
        im=Image.fromarray(ans.astype("uint8"))
        im2=Image.fromarray((ans*255).astype("uint8"))
        im.save(jpg_path)
        im2.save(visio_path)
        return

    def fcn_classcification_main(self):
        for i in tqdm(range(self.H)):
            for j in range(self.W):
                if self.data_array[i][j][1]==1:
                    npy_path=os.path.join(self.npy_dir,str(int(self.data_array[i][j][0]))+'.npy')
                    jpg_path=os.path.join(self.save_dir,str(int(self.data_array[i][j][0]))+'.jpg')
                    if os.path.exists(npy_path):
                        visio_path = os.path.join(self.visio_dir, str(int(self.data_array[i][j][0])) + '.jpg')
                        ans=self.fcn_classcification(npy_path)
                        self.save_ans(ans,jpg_path,visio_path)
                    else:
                        self.data_array[i][j][1] = 0
        print("FCN DONE!!...")
        return

if __name__ == '__main__':
    npy_file = "../pic_/pic.npy"
    data = np.load(npy_file)
    npy_dir="../cache/npy_dir"
    visio="../cache/visio/"
    save_dir="../cache/save_path"
    model_path="fcn_model/model/model1.h5"
    fcndeal=FCNDEAL(data,npy_dir,visio,save_dir,model_path)
    fcndeal.fcn_classcification_main()
    np.save("fcn",fcndeal.data_array)