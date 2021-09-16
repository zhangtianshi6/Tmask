import argparse
import torch
import utils
import os,cv2
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from collections import defaultdict


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

args_eval = parser.parse_args()
import modules_eval_transformer as modules

meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 50
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')
dataset_eval = utils.PathDataseth5seq1(hdf5_file=args.dataset.replace("train", "eval"), seq_len=args.seqences_len, path_length=1)
eval_loader = data.DataLoader(
    dataset_eval, batch_size=50, shuffle=False, num_workers=4)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0][0][0].size()

model_name = os.listdir(args_eval.save_folder)
max_h1score = -1
max_number = -1
record_name = ''
for name in model_name:
    if '.pt' in name and 'H1_' in name and '_MRR' in name:
        tmp = name.split("H1_")[1].split("_MRR")[0]
        print(tmp)
        if max_h1score<float(tmp):
            max_h1score = float(tmp)
            record_name = name
            max_number = int(name.split("epoch")[1].split("_H1")[0])
        if max_h1score==float(tmp):
            now_number = int(name.split("epoch")[1].split("_H1")[0])
            if now_number>max_number:
                max_h1score = float(tmp)
                record_name = name
                max_number = now_number

if ''!=record_name:
    model_name = record_name
else:
    model_name = 'model.pt'
print(model_name)

#model_name = "model_epoch108_H1_0.99047_MRR_tensor.pt"
print(model_name)
meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, model_name)

args = pickle.load(open(meta_file, 'rb'))['args']
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

topk = [1]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0
index_save = 0

pred_states = []
next_states = []


diff_data = []
save_num = 10

epoch_num = model_name.split('epoch')[1].split('_')[0]
save_pic_path1 = 'map/'+model_file.split('/')[-2].split('.')[0]
save_pic_path = save_pic_path1+'/img_eval1_'+epoch_num+'/'
if not os.path.exists(save_pic_path1):
    os.mkdir(save_pic_path1)
if not os.path.exists(save_pic_path):
    os.mkdir(save_pic_path)


with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [[t.to(
            device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch
        if batch_idx>=2:
            break
        obs = observations[0]
        next_obs = observations[-1]
        test_batch_size = obs.size(0)
        seq_num = obs.size(1)
        obj_c = obs.size(2)
        obj_h = obs.size(3)
        obj_w = obs.size(4)
        obs = obs.view(test_batch_size*seq_num, obj_c, obj_h, obj_w)
        next_obs = next_obs.view(test_batch_size*seq_num, obj_c, obj_h, obj_w)
        
        if index_save>=0 and index_save<10:
            np_obj = np.zeros((obs.size(2), obs.size(3), 3))
            for i in range(obs.size(0)):
                act = 0
                np_obj[:, :, 0] = obs[i, 0, :, :].cpu().detach().numpy()*255
                np_obj[:, :, 1] = obs[i, 1, :, :].cpu().detach().numpy()*255
                np_obj[:, :, 2] = obs[i, 2, :, :].cpu().detach().numpy()*255
                cv2.imwrite(save_pic_path+str(index_save)+'_'+str(i)+'_boj2_'+str(act)+'.jpg', np_obj)
                if next_obs.size(1)>3:
                    np_obj[:, :, 0] = obs[i, 3, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 1] = obs[i, 4, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 2] = obs[i, 5, :, :].cpu().detach().numpy()*255
                    cv2.imwrite(save_pic_path+str(index_save)+'_'+str(i)+'_boj1_'+str(act)+'.jpg', np_obj)
        if index_save>=0  and index_save<10:
            np_obj = np.zeros((next_obs.size(2), next_obs.size(3), 3))
            for i in range(obs.size(0)):
                act = 0
                np_obj[:, :, 0] = next_obs[i, 0, :, :].cpu().detach().numpy()*255
                np_obj[:, :, 1] = next_obs[i, 1, :, :].cpu().detach().numpy()*255
                np_obj[:, :, 2] = next_obs[i, 2, :, :].cpu().detach().numpy()*255
                cv2.imwrite(save_pic_path+str(index_save)+'_'+str(i)+'_nextobj2_'+str(act)+'.jpg', np_obj)

                if next_obs.size(1)>3:
                    np_obj[:, :, 0] = next_obs[i, 3, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 1] = next_obs[i, 4, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 2] = next_obs[i, 5, :, :].cpu().detach().numpy()*255
                    cv2.imwrite(save_pic_path+str(index_save)+'_'+str(i)+'_nextobj1_'+str(act)+'.jpg', np_obj)

        obj_state = model.obj_extractor(obs)
        state = model.obj_encoder(obj_state)
        objs_reshape = obj_state.view(test_batch_size*seq_num, num_obj, obj_h*obj_w)
        objs_reshape, indice = torch.sort(objs_reshape, -1)
        fenbu_loss = torch.zeros_like(objs_reshape).mean(2)
        zeros = torch.zeros_like(fenbu_loss).mean(1)
        for i in range(1, num_obj+1):
            befor_i = i -1
            i = i-num_obj+1 if i >num_obj-1 else i
            ob1 = objs_reshape[:, befor_i, :]
            ob2 = objs_reshape[:, i, :]
            cos = F.cosine_similarity(ob1, ob2, dim=-1)
            print(cos.size(), cos[0])

        obj_next_state = model.obj_extractor(next_obs)
        next_state = model.obj_encoder(obj_next_state)

        if index_save>=0 and index_save<10:
            for i in range(save_num):
                
                for k in range(obj_state.shape[1]):
                    np_obj = np.zeros((obj_state.size(2), obj_state.size(3), 3))
                    np_obj[:, :, 0] = obj_state[i, k, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 1] = obj_state[i, k, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 2] = obj_state[i, k, :, :].cpu().detach().numpy()*255
                    cv2.imwrite(save_pic_path+str(index_save)+'_'+str(i)+'_boj_'+str(k)+'.jpg', np_obj)
                    np_obj[:, :, 0] = obj_next_state[i, k, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 1] = obj_next_state[i, k, :, :].cpu().detach().numpy()*255
                    np_obj[:, :, 2] = obj_next_state[i, k, :, :].cpu().detach().numpy()*255
                    cv2.imwrite(save_pic_path+str(index_save)+'_'+str(i)+'_nextobj_'+str(k)+'.jpg', np_obj)

        pred_state = state
        obj_num = pred_state.size(1)
        pred_state = pred_state.view(test_batch_size, seq_num, obj_num, -1)
        next_state = next_state.view(test_batch_size, seq_num, obj_num, -1)
        for i in range(args_eval.num_steps):
            pred_trans = model.transition_model(pred_state, next_state)
            pred_trans = pred_trans[:, 0]
            pred_state = pred_state[:, 0]
            next_state = next_state[:, 0]
            pred_state = pred_state + pred_trans
            for k in range(len(actions[i])):
                act_data = 0
                np_obj = np.zeros((state[k].size(0), state[k].size(1)))
                np_obj = state[k].cpu().detach().numpy()

                np_obj = pred_trans[k].cpu().detach().numpy()
                np_obj = next_state[k].cpu().detach().numpy()


        index_save+=1
        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())
        
    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)

    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions
    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    dist_matrix = utils.pairwise_distance_matrix(
        next_state_flat, pred_state_flat)
    dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
    dist_matrix_augmented = torch.cat(
        [dist_matrix_diag, dist_matrix], dim=1)

    # Workaround to get a stable sort in numpy.
    dist_np = dist_matrix_augmented.numpy()
    indices = []
    for row in dist_np:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)
    indices = torch.from_numpy(indices).long()

    print('Processed {} batches of size {}'.format(
        batch_idx + 1, args.batch_size))

    labels = torch.zeros(
        indices.size(0), device=indices.device,
        dtype=torch.int64).unsqueeze(-1)

    num_samples += full_size
    print('Size of current topk evaluation batch: {}'.format(
        full_size))

    for k in topk:
        match = indices[:, :k] == labels
        num_matches = match.sum()
        hits_at[k] += num_matches.item()

    match = indices == labels
    _, ranks = match.max(1)

    reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
    rr_sum += reciprocal_ranks.sum()

    pred_states = []
    next_states = []

for k in topk:
    print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

print('MRR: {}'.format(rr_sum / float(num_samples)))

