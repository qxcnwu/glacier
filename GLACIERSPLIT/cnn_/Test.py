import os
from Efficinet import cnn_model
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def Model_acc(pic_dir="",txt_path="",model_path=""):
    model = cnn_model()
    model.load_weights(model_path)
    true=0
    flag=0

    files=open(txt_path)
    files=files.readlines()
    files=files[0:1000]
    for file in tqdm(files):
        true+=1
        jpg_path=os.path.join(pic_dir,file.split(",")[0])
        true_ans=int(file.split(",")[-1].replace("\n",""))
        ans=Compute_ans(jpg_path,model)
        if ans==true_ans:
            flag+=1
    return true,flag

def standard(data):
    """
    图像标准化
    :param data: 输入
    :return: 输出归一化数据
    """
    out_data = (data - np.mean(data)) / max(1 / 256, np.std(data))
    return out_data

def Compute_ans(pic_path,model):
    jpg=Image.open(pic_path)
    jpg_data=standard(jpg)
    jpg_data=np.reshape(jpg_data,(1,256,256,3))
    prediction = model.predict(jpg_data)
    ans = np.argmax(prediction, axis=-1)
    return ans


if __name__ == '__main__':
    pic_dir="../data/csave/"
    label_dir="../data/save_label/"
    txt_path="ans.txt"
    save_dir="../data/save/"
    model_path="model/model_cnn_2.h5"
    u,f=Model_acc(pic_dir,txt_path,model_path)