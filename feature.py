import numpy as np
from glob import glob
import datetime
import json
import os

yolo_debug = True

folders = [
    "data_converter/data_organized/feature/JMG1XXXXX",
    "data_converter/data_organized/feature/JMG4XXXXX",
    "data_converter/data_organized/feature/JMG5XXXXX",
    "data_converter/data_organized/feature/JMG6XXXXX",
    "data_converter/data_organized/feature/JMG7XXXXX",
    "data_converter/data_organized/feature/JMG8XXXXX"]

yolo_files = glob("data_converter/data_organized/yolo/*")

save_path = "data_converter/l2_out"

def nan_rate(array):
    nan_count = np.isnan(array).sum()
    total_count = 1
    for i in array.shape:
        total_count *= i
    nan_rate = nan_count / total_count
    return nan_rate

def convert_to_feature(dlc_root, yolo_root, save_dir, video_name, tstamp, frame_width = 640, frame_height = 480, mouse_l2_index = [0, 4], xy_pair = [[0,1],[10,11],[15,16],[20,21],[25,26],[30,31],[35,36],[40,41]], dim_xy_pair=[[2,3],[12,13],[17,18],[22,23],[27,28],[32,33],[37,38],[42,43]]):
    dlc_np = None
    yolo_np = None
    ls = None
    
    # ------------------- Load DeepLabCut data -------------------
    dlc_file = dlc_root+"/"+video_name.replace("_","/")+".npy"
    if not os.path.exists(dlc_file):
        print("File not found", dlc_file)
        return
    dlc_np = np.load(dlc_file)
    dlc_np = dlc_np.transpose(0, 2, 1)
    
    # ----- Remove cockroach data from DeepLabCut data -----
    dlc_np = np.delete(dlc_np, 6, axis=1)
    dlc_np = np.delete(dlc_np, 6, axis=1)
    
    # ------------------- Load Yolo data -------------------
    yolo_np = np.load(yolo_root)
    yolo_np = yolo_np.transpose(1, 0)
    temp_np = []
    for i in range(len(xy_pair)):
        __temp = [(yolo_np[xy_pair[i][0]]*frame_width+((yolo_np[dim_xy_pair[i][0]]*frame_width) if dim_xy_pair[i][0]>0 else (yolo_np[xy_pair[i][0]]*frame_width)))/2, (yolo_np[xy_pair[i][1]]*frame_width+((yolo_np[dim_xy_pair[i][1]]*frame_height/2) if dim_xy_pair[i][1]>0 else yolo_np[xy_pair[i][1]]*frame_width))/2]
        temp_np.append(__temp)
    # yolo_np = np.array([[yolo_np[0]*frame_width, yolo_np[1]*frame_height], [yolo_np[4]*frame_width, yolo_np[5]*frame_height], [yolo_np[8]*frame_width, yolo_np[9]*frame_height]])
    yolo_np = np.array(temp_np).transpose(2, 0, 1)
    yolo_np[yolo_np < 0] = np.nan
    yolo_np[yolo_np > 1000] = np.nan
    
    # ------------------ Check integrity -------------------
    if dlc_np.shape[0] != yolo_np.shape[0]:
        if yolo_debug:
            print(yolo_file.split("/")[-1], yolo_np.shape[0], dlc_np.shape[0])
            return
        else:
            raise ValueError("Yolo Frames and DeepLabCut Frames not equal")
        # dlc_np = dlc_np[:yolo_np.shape[0]]
    
    # ---------- Calculate L2 between body parts -----------
    local_np = dlc_np
    local_np = local_np.transpose(1, 0, 2)
    l2s = []
    for i in range(dlc_np.shape[1]):
        for j in range(dlc_np.shape[1]):
            if i >= j:
                continue
            dis_diff = local_np[i]-local_np[j]
            dis_diff = dis_diff**2
            dis_diff = np.sum(dis_diff, axis=1)
            dis_diff = np.sqrt(dis_diff)
            l2s.append(dis_diff)
    l2s = np.array(l2s)
    l2s = l2s.transpose(1, 0)
    ls = l2s
        
    # ---------- Calculate L2 between body parts and other objects -----------
    l2s = []
    local_yolo = yolo_np.transpose(1, 0, 2)
    local_dlc = dlc_np.transpose(1, 0, 2)
    for i in mouse_l2_index:
        for j in range(local_yolo.shape[0]):
            dis_diff = local_dlc[i]-local_yolo[j]
            dis_diff = dis_diff**2
            dis_diff = np.sum(dis_diff, axis=1)
            dis_diff = np.sqrt(dis_diff)
            l2s.append(dis_diff)
    l2s = np.array(l2s)
    l2s = l2s.transpose(1, 0)
    ls = np.concatenate((ls, l2s), axis=1)
    
    ls = np.concatenate((ls, dlc_np.reshape(-1, 12)), axis=1)
    
    for i in range(ls.shape[1]):
        tt = ls[..., i]
        if np.isnan(tt).all():
            continue
        lmax, lmin = np.percentile(tt[~np.isnan(tt)], [99.5, 0.5])
        if lmax == lmin:
            ls[..., i] = (ls[..., i] - lmin) / (lmax)
        else:
            ls[..., i] = (ls[..., i] - lmin) / (lmax - lmin)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.save(save_dir+"/"+video_name+"_"+current_time+"_l2.npy", ls)
    if nan_rate(ls) > 0.5:
        print(save_dir+"/"+video_name+"_"+current_time+"_l2.npy",nan_rate(ls),nan_rate(dlc_np))
    
def get_tstamps():
    tstasmps = {}
    for folder in folders:
        animal = folder.split("/")[-1]
        tstasmps[animal] = []
        for file in glob(folder+"/*.npy"):
            tstasmp = file.split("/")[-1].split(".")[0]
            if tstasmp not in tstasmps[animal]:
                tstasmps[animal].append(tstasmp)
    return tstasmps

def organize_yolo():
    organized = {}
    for yolo_file in yolo_files:
        video_name = yolo_file.split("/")[-1].split("_")[0]
        if video_name not in organized:
            organized[video_name] = []
        organized[video_name].append(yolo_file)
    return organized

def organize_yolo_w_tstamp():
    organized = {}
    for yolo_file in yolo_files:
        video_name = yolo_file.split("/")[-1].split("_")[0]
        tstmp = yolo_file.split("/")[-1].split("_")[1][3:]
        if video_name not in organized:
            organized[video_name] = {}
        if tstmp not in organized[video_name]:
            organized[video_name][tstmp] = []
        organized[video_name][tstmp].append(yolo_file)
    return organized

organized_yolo = organize_yolo_w_tstamp()

if not os.path.exists(save_path):
    os.makedirs(save_path)

for key in get_tstamps():
    for tstmp in get_tstamps()[key]:
        yolo_file = organized_yolo[key.split("X")[0]][tstmp]
        if len(yolo_file) > 1:
            raise ValueError("More than one yolo file")
        yolo_file = yolo_file[0]
        convert_to_feature('/'.join(folders[0].split("/")[:-1]), yolo_file, save_path, key+"_"+tstmp, tstmp)
