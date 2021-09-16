import argparse
import utils
import os, cv2
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import json

from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from itertools import groupby


def coco_eval_mask(annFile, resFile):
    annType = 'segm'
    annFile = annFile if annFile != '' else "tr_label.json"
    cocoGt=COCO(annFile)
    resFile = resFile if resFile != '' else "pr_label.json"
    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds
    imgId = imgIds[np.random.randint(len(imgIds))]
    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def mask_iou(masks_a, masks_b, iscrowd=False):
    
    masks_a = masks_a.view(masks_a.size(0), -1)
    masks_b = masks_b.view(masks_b.size(0), -1)

    intersection = masks_a @ masks_b.t()
    area_a = masks_a.sum(dim=1).unsqueeze(1)
    area_b = masks_b.sum(dim=1).unsqueeze(0)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def mask_binary(ground_truth_binary_mask):
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    rle = binary_mask_to_rle(fortran_ground_truth_binary_mask)
    compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    return compressed_rle

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

def write_intersection(img_ls, inters_ls, txt_name):
    fw = open(txt_name, 'w')
    if len(img_ls)!=len(inters_ls):
        print('len(img)!=len(inters).')
        return 0
    for i in range(len(img_ls)):
        fw.write(img_ls[i]+' ')
        print(inters_ls[i], i)
        for k in range(inters_ls[i].shape[0]):
            for j in range(inters_ls[i].shape[1]):
                fw.write(str(inters_ls[i][k, j])+' ')
        fw.write('\n')
    return 1

def write_json(img_ls, tr_masks, pr_masks):
    json_info = []
    json_trinfo = []
    for i in range(len(img_ls)):
        for k in range(pr_masks[i].shape[0]):
            tmp_info = {}
            tmp_info['image_id'] = int(img_ls[i].split('.')[0])
            tmp_info['category_id'] = 1
            tmp_info['score'] = 1.0
            tmp_info['segmentation'] = mask_binary(pr_masks[i][k])
            json_info.append(tmp_info)
            tmp_info = {}
            tmp_info['image_id'] = int(img_ls[i].split('.')[0])
            tmp_info['category_id'] = 1
            tmp_info['iscrowd'] = 0
            tmp_info['id'] = i*5+k
            tmp_info['score'] = 1.0
            tmp_info['area'] = np.sum(tr_masks[i][k])
            tmp_info['segmentation'] = mask_binary(tr_masks[i][k])
            json_trinfo.append(tmp_info)
    annFile = "instances_val2014.json"
    dataset = json.load(open(annFile, 'r'))
    dataset["annotations"] = json_trinfo
    dataset["images"] = images_info
    fw = open("tr_label.json", 'w')
    fw.write(json.dumps(dataset,cls=MyEncoder,indent=4))
    fw = open("pr_label.json", 'w')
    fw.write(json.dumps(json_info,cls=MyEncoder,indent=4))

def process_img(img):
    np_img = np.zeros((1, 3, img.shape[1], img.shape[0]))
    np_img[:, 0, :, :] = img[:,:,0]
    np_img[:, 1, :, :] = img[:,:,1]
    np_img[:, 2, :, :] = img[:,:,2]
    np_img = np.float32(np_img/255.)
    tor_arr= torch.from_numpy(np_img)
    tor_arr = tor_arr.to(device)
    return tor_arr


torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--choice-model', type=str, default="",
                    help='Dimensionality of heads space.')
parser.add_argument('--model-name', type=str, default='model.pt',
                    help='Path to model.')
args_eval = parser.parse_args()
import modules_eval_transformer as modules

save_folder = args_eval.save_folder
model_name = args_eval.model_name
meta_file = save_folder+'/'+'metadata.pkl'
model_file = save_folder+'/'+model_name
print('model_file', model_file)
data_select = 'cube'
if 'shape' in model_file:
    data_select = 'shape'

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 50
args.dataset = args_eval.dataset
args.seed = 0
args.action_dim = 20

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset_eval = utils.PathDataset(
    hdf5_file=args.dataset.replace("train", "eval"), path_length=1)
eval_loader = data.DataLoader(
    dataset_eval, batch_size=50, shuffle=False, num_workers=4)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0][0].size()

meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, model_name)

args = pickle.load(open(meta_file, 'rb'))['args']
device = torch.device('cuda' if args.cuda else 'cpu')

num_obj = args.num_objects
model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    encoder=args.encoder,
    heads = args.heads,
    layers = args.layers, 
    trans_model = args.choice_model).to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

img_path = "/home/zt/Share1/workspace/c-swm1/c-swm/data/eval_data/"+data_select+"/roi/"
img_ls = os.listdir(img_path)
tr_masks = np.zeros((len(img_ls), num_obj, input_shape[1], input_shape[2]))

pr_thre = 0.7
images_info = []
fw_intersection_ls = []
fw_img_ls = []
pr_masks = []
tr_masks = []
with torch.no_grad():
    for i in range(len(img_ls)):
        # read true img mask
        img_name = img_path+img_ls[i]
        img = cv2.imread(img_name)
        masks = []
        tr_mask = np.zeros((num_obj, input_shape[1], input_shape[2]))
        pr_mask = np.zeros((num_obj, input_shape[1], input_shape[2]))
        for k in range(num_obj):
            mask_pic = cv2.imread(img_name.replace("roi", "mask").replace(".png", "_"+str(k)+".png"), 0)
            tr_mask[k, :, :] = mask_pic/255
            tr_mask[k][tr_mask[k] >= pr_thre] = 1
            tr_mask[k][tr_mask[k] <  pr_thre] = 0
            
        # forward
        obj_state = model.obj_extractor(process_img(img))
        for k in range(obj_state.size(1)):
            mask_pic = obj_state[0, k, :,:].cpu().detach().numpy()
            mask_pic[mask_pic >= pr_thre] = 1
            mask_pic[mask_pic < pr_thre] = 0 
            pr_mask[k, :, :] = mask_pic
        tr_masks.append(tr_mask)
        pr_masks.append(pr_mask)
        images_info.append({'file_name': img_name, 'height': input_shape[2], 'license': 1, 'flickr_url': '', 'width': input_shape[1], 'id': int(img_ls[i].split('.')[0]), 'date_captured': '', 'coco_url': ''})
        intersection = mask_iou(torch.from_numpy(tr_mask), torch.from_numpy(pr_mask))
        intersection = intersection.cpu().detach().numpy()
        fw_intersection_ls.append(intersection)
        fw_img_ls.append(img_ls[i])
        
#write_intersection(fw_img_ls, fw_intersection_ls, "cube_output.txt")
write_json(fw_img_ls, tr_masks, pr_masks)
print(len(tr_masks), tr_masks[0].shape, len(pr_masks), pr_masks[0].shape)
coco_eval_mask("tr_label.json", "pr_label.json")





        










