import os

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
    return

def del_main(cut_dir,npy_dir,save_path,visio):
    print("INIT START!!...")
    del_files(cut_dir)
    del_files(npy_dir)
    del_files(save_path)
    del_files(visio)
    print("INIT DONE!!...")
    return
