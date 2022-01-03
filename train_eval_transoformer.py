import argparse
import torch
import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data
import torch.nn.functional as F
from collections import defaultdict

#import modules

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')
parser.add_argument('--encoder', type=str, default='small',
                    help='Object extrator CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5', help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--heads', type=int, default=8,
                    help='Dimensionality of heads space.')
parser.add_argument('--layers', type=int, default=2,
                    help='Dimensionality of heads space.')
parser.add_argument('--seqences-len', type=int, default=10,
                    help='Dimensionality of seq. len.')

parser.add_argument('--trained-model', type=str, default="",
                    help='Dimensionality of heads space.')
parser.add_argument('--choice-model', type=str, default="",
                    help='Dimensionality of heads space.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

import modules_eval_transformer as modules


now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
checkpoints = "checkpoints"
exp_counter = 0
save_folder = checkpoints+'/'+exp_name#'{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')

pickle.dump({'args': args}, open(meta_file, "wb"))
device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.StateTransitionsDataseth5seq1(hdf5_file=args.dataset, seq_len=args.seqences_len)
train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

#dataset_eval = utils.PathDataseth5seq1(hdf5_file=args.dataset.replace("train", "eval"), seq_len=args.seqences_len, path_length=1)
dataset_eval = utils.StateTransitionsDataseth5seq1(hdf5_file=args.dataset.replace("train", "eval"), seq_len=args.seqences_len)
eval_loader = data.DataLoader(dataset_eval, batch_size=50, shuffle=False, num_workers=4) # 50

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0][0].size()

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

start_epoch = 0
if args.trained_model!="":
    model.load_state_dict(torch.load(args.trained_model))
    start_epoch = int(args.trained_model.split('epoch')[1].split('_')[0])
else:
    model.apply(utils.weights_init)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate)
model = torch.nn.DataParallel(model)


# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9
index_save = 0

for epoch in range(start_epoch, args.epochs + 1):
    model.train()
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()
        #loss, ps_loss, neg_loss, black_loss= model.contrastive_loss(*data_batch)
        loss, ps_loss, neg_loss, black_loss, next_state, pred_trans= model(*data_batch)
        loss = loss.mean()
        ps_loss = ps_loss.mean()
        neg_loss = neg_loss.mean()
        black_loss = black_loss.mean()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} pos_loss: {:.6f} neg_loss: {:.6f} b_loss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0]), ps_loss.item() / len(data_batch[0]), neg_loss.item() / len(data_batch[0]), black_loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss or epoch%100==0:
        best_loss = avg_loss
        
        # eval
        topk = [1]
        hits_at = defaultdict(int)
        num_samples = 0
        rr_sum = 0
        index_save = 0

        pred_states = []
        next_states = []
        model.eval()

        with torch.no_grad():
            for batch_idx, data_batch in enumerate(eval_loader):
                #data_batch = [[t.to(
                #    device) for t in tensor] for tensor in data_batch]
                data_batch = [tensor.to(device) for tensor in data_batch]
                # observations, actions = data_batch

                if batch_idx>20:
                    break
		
                obs, action, next_obs, obs_neg, action_neg, next_obs_neg = data_batch
                loss, ps_loss, neg_loss, black_loss, next_state, pred_state= model(*data_batch)

                test_batch_size = obs.size(0)
                seq_num = obs.size(1)
                obj_num = pred_state.size(2)
                pred_state = pred_state.view(test_batch_size, seq_num, obj_num, -1)
                next_state = next_state.view(test_batch_size, seq_num, obj_num, -1)
                
                for i in range(1):
                    pred_state = pred_state[:, 0]
                    next_state = next_state[:, 0]

                pred_states.append(pred_state.cpu())
                next_states.append(next_state.cpu())

            pred_state_cat = torch.cat(pred_states, dim=0)
            next_state_cat = torch.cat(next_states, dim=0)

            full_size = pred_state_cat.size(0)

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

        h1 = hits_at[k]*1.0/float(num_samples)
        print('MRR: {}'.format(rr_sum / float(num_samples)))
        MRR = rr_sum *1.0 / float(num_samples)
        model_file = os.path.join(save_folder, 'model.pt')
        model_file = model_file.split('.')[0]+'_epoch'+str(epoch)+'_H1_'+str(h1)[:7]+'_MRR_'+str(MRR)[:6]+'.pt'
        torch.save(model.module.cpu().state_dict(), model_file)
        model = model.to(device)
        
        print('save model', model_file)
