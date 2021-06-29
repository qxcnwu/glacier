import os
import numpy as np
from main import cut
from PIL import Image
from tqdm import tqdm

def made_txt(jpg_dir,save_dir,txt_path):
    files=os.listdir(jpg_dir)
    k=os.listdir(save_dir)
    fd=open(txt_path,"w")
    for file in tqdm(files):
        jpg_path=os.path.join(jpg_dir,file)
        im=np.array(Image.open(jpg_path))
        if np.max(im)<=10:
            os.remove(jpg_path)
            continue
        if file not in k:
            flag=0
        else:
            flag=1
        fd.write(file+","+str(flag)+"\n")
    fd.close()
    return

if __name__ == '__main__':
    jpg_path="C:/Users/86188/PycharmProjects/MAP/glacier/data/jpg"
    label_path=""
    size=256
    jpg_save_dir="../data/csave/"
    # cut(jpg_path,"",jpg_save_dir,"",256)
    made_txt(jpg_save_dir,"../data/save_jpg/","ans.txt")