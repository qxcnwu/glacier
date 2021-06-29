import numpy as np
import os
from tqdm import tqdm
# from Efficinet import cnn_model

import tensorflow.keras as keras
import efficientnet.tfkeras as efn

def cnn_model():
    backbone = efn.EfficientNetB4(input_shape=(256, 256, 3), weights=None, include_top=False)
    input = backbone.input
    x = keras.layers.GlobalAveragePooling2D()(backbone.output)
    output = keras.layers.Dense(2, activation=None)(x)
    model = keras.models.Model(inputs=input, outputs=output)
    return model

class CNNDEAL:
    def __init__(self,model_path:os.path,data_array:np.array,npy_dir:os.path):
        """
        CNN处理文件矩阵
        :param model_path: 模型路径
        :param data_array: 文件矩阵
        :param npy_dir: npy文件路径
        """
        print("PIC CNNING !!...")
        self.data_array=data_array
        self.npy_dir=npy_dir
        self.H,self.W,_=self.data_array.shape
        self.model = cnn_model()
        self.model.load_weights(model_path)

    def classification(self,npy_path):
        """
        读取npy并且分类
        :param npy_path: npy文件路径
        :return:
        """
        data=np.load(npy_path)
        data=np.expand_dims(data,axis=0)
        prediction = self.model.predict(data)
        ans = np.argmax(prediction, axis=-1)
        return ans

    def classification_main(self):
        """
        CNN分类主程序
        :return:
        """
        for i in tqdm(range(self.H)):
            for j in range(self.W):
                if self.data_array[i][j][1]==1:
                    npy_path=os.path.join(self.npy_dir,str(int(self.data_array[i][j][0]))+".npy")
                    self.data_array[i][j][1]=self.classification(npy_path)
        print("CNN DONE!!...")
        return


    def data_fill(self,flag):
        """
        填充文件矩阵
        :param flag: 限制阈值
        :return:
        """
        print("FILLING PICTURE!!...")
        new_data=np.zeros((self.H,self.W))
        for i in range(1,self.H-1):
            for j in range(1,self.W-1):
                if self.data_array[i][j][1]==1 or np.sum(self.data_array[i-5:i+5,j-5:j+5,1])>=flag:
                    new_data[i][j]=1
        self.data_array[:,:,1]=new_data
        print("FILLING PICTURE DONE!!...")
        return self.data_array

if __name__ == '__main__':
    npy_file="pic.npy"
    data=np.load(npy_file)
    model_path="cnn_model/model_cnn_2.h5"
    npy_dir="../cache/npy_dir/"
    cnndeal=CNNDEAL(model_path,data,npy_dir)
    cnndeal.classification_main()
    cnndeal.data_fill(1)
    array=cnndeal.data_array
    np.save("cnn_fill",array)