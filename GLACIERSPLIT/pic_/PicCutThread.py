import cv2 as cv
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import threading
from PIL import Image

all(["PIC_CUT","PICPREDEAL","PICCLEAN","PICTONPY"])

class PICCUT:
    def __init__(self, pic_path: os.path, save_dir: os.path, cut_size=256, H=None, W=None):
        """
        图像多线程切割类
        :param pic_path: 图像路径
        :param save_dir: 保存路径
        :param cut_size: 剪裁大小
        """
        print("PIC CUTTING!!...")
        self.pic_path = pic_path
        self.save_dir = save_dir
        self.cut_size = cut_size
        self.H = H
        self.W = W
        self.img = cv.imread(self.pic_path)
        print(self.img.shape)
        if self.H == None or self.W == None:
            raise ValueError("H,W should be int not None!!")

    def pic_cut_(self, h, flag):
        """
        单线程切割函数
        :param h: 行号
        :param flag: 编号
        :return: NULL
        """
        for j in range(self.W):
            data = self.img[h * self.cut_size:(h + 1) * self.cut_size, j * self.cut_size:(j + 1) * self.cut_size]
            save_path = os.path.join(self.save_dir, str(flag) + ".jpg")
            flag += 1
            cv.imwrite(save_path, data)
        return

    def threading_cut(self, flag):
        """
        多线程切割
        :param flag: 初始编号
        :return: NULL
        """
        threads = []
        for i in tqdm(range(self.H)):
            t = threading.Thread(target=self.pic_cut_, args=(i, i * self.H + flag))
            threads.append(t)
        for i in range(self.H):
            threads[i].start()
        for i in range(self.H):
            threads[i].join()
        print("PIC CUT END!!...")
        return


class PICPREDEAL:
    def __init__(self, pic_path: os.path, save_path=None, type=".jpg", reshape=True, remove=False, cut_size=256):
        """
        tiff预处理 包括修改形状 格式转换
        :param pic_path:图像路径
        :param save_path:图像保存路径
        :param type:图像保存格式
        :param reshape:是否修改大小
        :param remove:是否删除源文件
        :param cut_size:调整像元
        """
        print("PIC PRE DEALING!!...")
        self.pic_path = pic_path
        if not save_path:
            self.save_path = pic_path.replace(".tif", type)
        else:
            self.save_path = save_path
        self.cut_size = cut_size
        self.reshape = reshape
        self.remove = remove
        self.W, self.H = self.pic_tran()
        self.array=self.pic_array()
        print("PIC PRE DEAL END!!...")

    def pic_tran(self):
        im = cv.imread(self.pic_path)
        H, W, C = im.shape
        if self.reshape:
            H = ((H // self.cut_size) + 1) * self.cut_size
            W = ((W // self.cut_size) + 1) * self.cut_size
            im = cv.resize(im, (H, W))
        cv.imwrite(self.save_path, im)
        if self.remove:
            os.remove(self.pic_path)
        return H // self.cut_size, W // self.cut_size

    def pic_array(self):
        data=np.zeros((self.H,self.W,2))
        for i in range(self.H):
            for j in range(self.W):
                data[i][j][0]=i*self.H+j
        return data

class PICCLEAN:
    def __init__(self,save_dir,data_array:np.array,limit=0.8):
        """
        图像清洗
        :param save_dir: 保存路径
        :param data_array: 文件矩阵
        :param limit: 限制条件
        """
        print("PIC CLEANING !!...")
        self.save_dir=save_dir
        self.data_array=data_array
        self.H,self.W,_=self.data_array.shape
        self.limit=limit
        self.scale=256

    def clean_main(self,path)->bool:
        """
        对单独文件判断其是否符合入册标准
        :param path: 路径
        :return: BOOL
        """
        im=np.array(Image.open(path))
        if np.max(im)==0 and len(np.where(im[:,:,0]==0)[0])/self.scale/self.scale>=self.limit:
            return False
        else:
            return True

    def clean_way(self,H):
        """
        单线程清理
        :param H: 当前索引
        :return: null
        """
        for i in range(self.W):
            path = os.path.join(self.save_dir, str(int(self.data_array[H][i][0])) + ".jpg")
            if self.clean_main(path):
                self.data_array[H][i][1]=1
        return

    def clean_threading(self):
        """
        多线程清理
        :return:
        """
        threads = []
        for i in tqdm(range(self.H)):
            t = threading.Thread(target=self.clean_way, args=(i,))
            threads.append(t)
        for i in range(self.H):
            threads[i].start()
        for i in range(self.H):
            threads[i].join()
        print("PIC CLEAN END!!...")
        return


class PICTONPY:
    def __init__(self,data_array:np.array,jpg_dir:os.path,npy_dir:os.path):
        """
        将图像转换为NPY文件便于读取
        :param data_array:文件矩阵
        :param npy_dir:npy保存路径
        :param jpg_dir:jpg保存路径
        """
        print("PIC TO NPY!!...")
        self.data_array=data_array
        self.npy_dir=npy_dir
        self.jpg_dir=jpg_dir
        self.H, self.W, _ = self.data_array.shape

    def cpu_standard(self,data):
        """
        numpy图像标准化
        :param data: 输入图像
        :return: 输出归一化数据
        """
        out_data = (data - np.mean(data)) / max(1 / 256, np.std(data))
        return out_data

    def gpu_standard(self,data):
        """
        cupy图像标准化
        :param data: 输入图像
        :return: 输出归一化数据
        """
        div=max(1/256,cp.std(data))
        out_data=cp.divide(cp.subtract(data,cp.mean(data)),div)
        return out_data

    def tran_main(self, jpg_path,npy_path):
        """
        单个文件转换npy文件
        :param jpg_path:
        :param npy_path:
        :return:
        """
        # cp.save(npy_path,self.gpu_standard(cp.asarray(np.array(Image.open(jpg_path)))))
        data=self.cpu_standard(np.array(Image.open(jpg_path)))
        np.save(npy_path,data)
        return

    def tran_way(self, H):
        """
        单线程转换
        :param H: 当前索引
        :return: null
        """
        for i in range(self.W):
            jpg_path = os.path.join(self.jpg_dir, str(int(self.data_array[H][i][0])) + ".jpg")
            npy_path = os.path.join(self.npy_dir, str(int(self.data_array[H][i][0])))
            if self.data_array[H][i][1]:
                self.tran_main(jpg_path,npy_path)
        return

    def tran_threading(self):
        """
        多线程转换
        :return:
        """
        threads = []
        for i in tqdm(range(self.H)):
            t = threading.Thread(target=self.tran_way, args=(i,))
            threads.append(t)
        for i in range(self.H):
            threads[i].start()
        for i in range(self.H):
            threads[i].join()
        print("PIC TO NPY END!!...")
        return

if __name__ == '__main__':
    """
    TEST
    """
    pic_path = "../cache/tiff/232.jpg"
    reshape = False
    remove = False
    save_dir = "../cache/cut_dir/"
    npy_dir="../cache/npy_dir/"
    picpreseal = PICPREDEAL(pic_path,reshape=reshape)
    H, W ,array= picpreseal.H, picpreseal.W,picpreseal.array
    piccut=PICCUT(pic_path,save_dir,H=H,W=W)
    piccut.threading_cut(0)
    picclean=PICCLEAN(save_dir,array)
    picclean.clean_threading()
    array=picclean.data_array
    pictonpy=PICTONPY(array,save_dir,npy_dir)
    pictonpy.tran_threading()

    np.save("pic",array)