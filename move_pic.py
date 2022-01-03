import cv2, os, shutil
img_path = "map/shape_obj5_trans_layers2_seq10_subsatall/"
f_ls = os.listdir(img_path)
for i in range(len(f_ls)):
    f_name = f_ls[i]
    img_ls = os.listdir(img_path+f_name)
    #for k in range(len())
    name0 = "0_0_boj2_0.jpg"
    name3 = "0_0_nextobj2_0.jpg"
    shutil.copy(img_path+f_name+'/'+name0, img_path+f_name+'_'+name0)
    shutil.copy(img_path+f_name+'/'+name3, img_path+f_name+'_'+name3)
    for k in range(5):
        name1 = "0_0_boj_"+str(k)+".jpg"
        name2 = "0_0_nextobj_"+str(k)+".jpg"
        shutil.copy(img_path+f_name+'/'+name1, img_path+f_name+'_'+name1)
        shutil.copy(img_path+f_name+'/'+name2, img_path+f_name+'_'+name2)
