import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
from PIL import Image

class PICMERGE:
    def __init__(self,data_array:np.array,save_dir:os.path,save_path:os.path,visio_path:os.path,cut_size=256):
        """
        合并结果图像
        :param data_array:文件矩阵
        :param save_dir: 结果保存
        :param save_path: 图像保存位置
        :param visio_path: 外显图像保存位置
        :param cut_size: 裁剪像元大小
        """
        print("MERGEING!!...")
        self.data_array=data_array
        self.save_dir=save_dir
        self.save_path=save_path
        self.visio_path=visio_path
        self.cut_size=cut_size
        self.H,self.W,_=self.data_array.shape
        self.ans=np.zeros((self.H*self.cut_size,self.W*self.cut_size))
        self.fill_=np.ones((self.cut_size,self.cut_size))
        self.fill=np.zeros((self.cut_size,self.cut_size))
        self.merge_main()

    def merge_main(self):
        for i in tqdm(range(self.H)):
            for j in range(self.W):
                if self.data_array[i][j][1]==0:
                    self.ans[i*self.cut_size:(i+1)*self.cut_size,j*self.cut_size:(j+1)*self.cut_size]=self.fill
                else:
                    jpg_path=os.path.join(self.save_dir,str(int(self.data_array[i][j][0]))+".jpg")
                    if os.path.exists(jpg_path):
                        self.ans[i * self.cut_size:(i + 1) * self.cut_size, j * self.cut_size:(j + 1) * self.cut_size]=np.array(Image.open(jpg_path))
        im=Image.fromarray(self.ans.astype("uint8"))
        im.save(self.save_path)
        print("MERGE DONE!!...")
        return

if __name__ == '__main__':
    npy_file = "../pic_/pic.npy"
    data = np.load(npy_file)
    save_dir="../cache/visio/"
    save_path="1.jpg"
    visio="2.jpg"
    PICMERGE(data,save_dir,save_path,visio)