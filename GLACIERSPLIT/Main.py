from pic_.PicCutThread import PICCUT,PICCLEAN,PICTONPY,PICPREDEAL
from fcn_.FcnClassFication import FCNDEAL
from cnn_.CnnClassFication import CNNDEAL
from pic_.PICMERGE import PICMERGE
from CLEANFILE import del_main
import warnings
import os,sys
sys.path.append("/GLACIERSPLIT/")
# 配置
warnings.filterwarnings("ignore")

def main(pic_path="",save_dir="",cut_dir="",npy_dir="",cnn_model="",visio_dir="",fcn_model="",ans_path=""):
    # 初始化
    del_main(save_dir,npy_dir,cut_dir,visio_dir)
    # 预处理
    picpreseal = PICPREDEAL(pic_path,reshape=True,remove=False)
    H, W, array = picpreseal.H, picpreseal.W, picpreseal.array
    # 剪裁
    piccut = PICCUT(pic_path, cut_dir, H=H, W=W)
    piccut.threading_cut(0)
    # 清洗
    picclean = PICCLEAN(cut_dir, array)
    picclean.clean_threading()
    array = picclean.data_array
    # 转换
    pictonpy = PICTONPY(array, cut_dir, npy_dir)
    pictonpy.tran_threading()

    # # CNN分类
    # cnndeal = CNNDEAL(cnn_model, array, npy_dir)
    # cnndeal.classification_main()
    # cnndeal.data_fill(1)
    # array = cnndeal.data_array

    # fcn预测
    fcndeal = FCNDEAL(array, npy_dir, visio_dir, save_dir, fcn_model)
    fcndeal.fcn_classcification_main()

    # 合并
    PICMERGE(array, visio_dir, ans_path, ans_path)

    return

if __name__ == '__main__':
    pic_dir= "cache/tiff"
    save_dir= "cache/save_path"
    cut_dir= "cache/cut_dir"
    npy_dir= "cache/npy_dir"
    cnn_model="cnn_/cnn_model/model_cnn_2.h5"
    visio_dir= "cache/visio"
    fcn_model="fcn_/fcn_model/model/model1.h5"
    ans_dir= "cache/ans"

    files=os.listdir(pic_dir)
    for file in files:
        pic_path=os.path.join(pic_dir,file)
        ans_path=os.path.join(ans_dir,file)
        main(pic_path,save_dir,cut_dir,npy_dir,cnn_model,visio_dir,fcn_model,ans_path)