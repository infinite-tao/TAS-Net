from __future__ import print_function
import torch
import os.path as osp
import time
import argparse
import datetime
import os
import torch.nn as nn
import h5py
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from models import DSN, FC
from utils import weights_init, save_checkpoint, inv_lr_scheduler, read_json
from rewards import compute_reward_det_coff, compute_reward_coff, compute_reward
import numpy as np
import random
from scipy.io import savemat
from Graph_Net import ClassifierGNN
from evaluate import evaluate
import math

# Parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', default=False, help='Training or Validate.')
parser.add_argument('--seed', type=int, default=27, help='Random seed')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--subject_id', type=int, default=0, help="subject index (default: 0)")
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--edge_features', type=int, default=32, help='graph edge features dimension.')
parser.add_argument('--n_feature', type=int, default=192, help='Number of hidden units.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hid_dim', type=int, default=256, help='hidden unit dimension of DSN (default: 256).')
parser.add_argument('--deep_features', type=str, default='./features/SEED/session_1/source_h5_file.h5', help='output directory and fragments')
# parser.add_argument('--label', type=str, default='./features/SEED/label.txt', help='emotion_localization_label')
# parser.add_argument('--save_path', type=str, default='/home/ubuntu/zhangyongtao/PycharmProjects/EEGfusenet/UEL-DRL/SEED/session_1/model/FC_layer', help='output directory and fragments')
parser.add_argument('--save_path', type=str, default='/home/ubuntu/zhangyongtao/PycharmProjects/TAS-Net/TAS-output/SEED/session1', help='output directory and fragments')
parser.add_argument('--fragment_length', type=int, default=8, help='Left or Right Maximum offset.')
parser.add_argument('--num_fragment', type=int, default=10, help='for eval emotion localization and unsupervised clustering.')
parser.add_argument('--reward_function', type=str, default='R1_R2', help='sim:R1, rep:R2 or mix:R1_R2.')
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use.")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if args.training:
        datasets = h5py.File(args.deep_features, 'r')
        num_videos = len(datasets.keys())
        all_keys = list(datasets.keys())
        test_keys = all_keys[15*args.subject_id:15*(args.subject_id+1)]
        del all_keys[15*args.subject_id:15*(args.subject_id+1)]
        train_keys = all_keys
        print("# total videos {}. # train videos {}. # test videos {}.".format(num_videos, len(train_keys),
                                                                               len(test_keys)))
        optimizer_config = {
            'type': torch.optim.Adam,
            'optim_params': {
                'lr': 0.001,
                'weight_decay': 0.0001,
            },
            'lr_type': 'inv',
            'lr_param': {
                'lr': 0.001,
                'gamma': 0.001,
                'power': 0.75,
            },
        }
        Network1 = ClassifierGNN(in_features=args.n_feature,
                                 edge_features=args.edge_features,
                                 out_features=args.n_feature,
                                 device=DEVICE)
        Network2 = DSN(in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru')
        # Network2 = FC(in_dim=192)
        Network2.apply(weights_init)
        print("Model size: {:.5f}M".format(sum(p.numel() for p in Network1.parameters()) / 1000000.0))
        print("Model size: {:.5f}M".format(sum(p.numel() for p in Network2.parameters()) / 1000000.0))
        parameter_list = [{'params': Network1.parameters(), 'lr_mult': 10, 'decay_mult': 2}] + [
            {'params': Network2.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
        optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

        param_lr = []
        for param_group in optimizer.param_groups:
            param_lr.append(param_group['lr'])
        schedule_param = optimizer_config['lr_param']

        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        start_epoch = 0
        Network1 = Network1.to(DEVICE)
        Network2 = Network2.to(DEVICE)

        print("=====> Start DRL training and Online Evaluate <===== ")
        start_time = time.time()
        Network1.train()
        Network2.train()

        baselines = {key: 0. for key in train_keys}  # baseline rewards for videos
        reward_writers = {key: [] for key in train_keys}  # record reward changes for each video
        best_recall = 0.
        # Inconsistency_loss = torch.nn.MSELoss(reduction='mean')
        for epoch in range(start_epoch, args.epochs):
            # optimizer = inv_lr_scheduler(optimizer, epoch, **schedule_param)

            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs)  # shuffle indices

            for idx in idxs:
                key = train_keys[idx]
                seq = datasets[key]['features'][...]

                seq = torch.from_numpy(seq)  # input shape (seq_len, dim)
                seq = seq.to(DEVICE)

                local_graphs0 = None
                for n in range(math.ceil(seq.shape[0] / (args.fragment_length * 2))):
                    if n != math.ceil(seq.shape[0] / (args.fragment_length * 2)) - 1:
                        data0 = seq[(args.fragment_length * 2) * n:(args.fragment_length * 2) * (n + 1), :]
                        sub_graph0, _ = Network1(data0)
                    else:
                        data0 = seq[(args.fragment_length * 2) * n:, :]
                        if data0.shape[0] == 1:
                            sub_graph0 = data0
                        else:
                            sub_graph0, _ = Network1(data0)

                    if local_graphs0 is not None:
                        local_graphs0 = torch.cat((local_graphs0, sub_graph0), dim=0)
                    else:
                        local_graphs0 = sub_graph0
                # #
                seq_graph0 = torch.add(seq, local_graphs0)
                seq = seq_graph0.unsqueeze(dim=0)
                sig_probs = Network2(seq)

                cost = 0.01 * (sig_probs.mean() - 0.5) ** 2
                # cost = Inconsistency_loss(sig_probs, sig_trans)
                m = Bernoulli(sig_probs)
                epis_rewards = []

                for episode in range(5):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    reward = compute_reward(seq, actions)
                    expected_reward = log_probs.mean() * (reward - baselines[key])
                    cost -= expected_reward  # minimize negative expected reward
                    epis_rewards.append(reward.item())

                baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(
                    epis_rewards)  # update baseline reward via moving average
                reward_writers[key].append(np.mean(epis_rewards))

                optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(Network1.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(Network2.parameters(), 5.0)
                optimizer.step()

            epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])

            if (epoch + 1) % 1 == 0:
                print("epoch {}/{}\t reward {}\t  cost {}\t".format(epoch + 1,
                                                                    100, epoch_reward, cost))
            if (epoch + 1) % 1 == 0:
                Recall = evaluate(args, Network1, Network2, datasets, test_keys)
                print(Recall)
                if Recall > best_recall:
                    best_recall = Recall
                    model_state_dict1 = Network1.state_dict()
                    model_state_dict2 = Network2.state_dict()
                    model_save_path1 = osp.join(args.save_path,
                                                'pretrained_best_model1_seed_' + 'subject' + str(args.subject_id) + '_' + str(args.reward_function) + '.pth.tar')
                    model_save_path2 = osp.join(args.save_path,
                                                'pretrained_best_model2_seed_' + 'subject' + str(args.subject_id) + '_' + str(args.reward_function) + '.pth.tar')
                    save_checkpoint(model_state_dict1, model_save_path1)
                    save_checkpoint(model_state_dict2, model_save_path2)
                    print("Model saved to {}".format(model_save_path1))
                    print("Model saved to {}".format(model_save_path2))

            Network1.train()
            Network2.train()
            scheduler.step()

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        model_state_dict1 = Network1.state_dict()
        model_state_dict2 = Network2.state_dict()
        model_save_path1 = osp.join(args.save_path, 'pretrained_model1_seed_' + str(args.reward_function) + '.pth.tar')
        model_save_path2 = osp.join(args.save_path, 'pretrained_model2_seed_' + str(args.reward_function) + '.pth.tar')
        save_checkpoint(model_state_dict1, model_save_path1)
        save_checkpoint(model_state_dict2, model_save_path2)
        print("Model saved to {}".format(model_save_path1))
        print("Model saved to {}".format(model_save_path2))

    else:
        datasets = h5py.File(args.deep_features, 'r')
        all_keys = list(datasets.keys())
        test_keys = all_keys[15 * args.subject_id:15 * (args.subject_id + 1)]

        print("# test videos {}.".format(len(test_keys)))
        model1 = ClassifierGNN(in_features=args.n_feature,
                                 edge_features=args.edge_features,
                                 out_features=args.n_feature,
                                 device=DEVICE)
        model2 = DSN(in_dim=args.n_feature, hid_dim=args.hid_dim, num_layers=1, cell='gru')
        # model2 = FC(in_dim=192)
        checkpoint_path1 = osp.join(args.save_path, 'pretrained_best_model1_seed_' + 'subject' + str(args.subject_id) + '_' + str(args.reward_function) + '.pth.tar')
        checkpoint1 = torch.load(checkpoint_path1)
        model1.load_state_dict(checkpoint1)
        model1 = model1.cuda()
        checkpoint_path2 = osp.join(args.save_path, 'pretrained_best_model2_seed_' + 'subject' + str(args.subject_id) + '_' + str(args.reward_function) + '.pth.tar')
        checkpoint2 = torch.load(checkpoint_path2)
        model2.load_state_dict(checkpoint2)
        model2 = model2.cuda()
        with torch.no_grad():
            model1.eval()
            model2.eval()
            out_path = os.path.join(args.save_path, 'result_output')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            save_idx = open(os.path.join(out_path, 'log_' + 'subject' + str(args.subject_id) + '_' + str(args.reward_function) + '_' + str(args.num_fragment) + '.txt'), 'w')
            all_features = None
            all_labels = None
            num_segments_trial = []
            local_labels = [[[13, 22], [204, 218], [219, 235]],
                 [[50, 65], [132, 150], [164, 187], [206, 226]],
                 [[14, 27], [66, 80], [135, 149], [150, 165], [186, 206]],
                 [[4, 24], [26, 45], [95, 121], [131, 136], [166, 183], [202, 212]],
                 [[15, 35], [35, 50], [135, 150]],
                 [[10, 19], [40, 49], [63, 74], [91, 103], [120, 129], [165, 181]],
                 [[23, 40], [61, 75], [152, 165], [180, 195], [200, 212]],
                 [],
                 [[55, 70], [128, 143], [165, 180], [215, 235]],
                 [[14, 34], [58, 83], [98, 108], [141, 151]],
                 [],
                 [[45, 63], [76, 91], [148, 159], [188, 204], [209, 219], [229, 240]],
                 [[21, 31], [92, 103], [119, 129], [224, 240]],
                 [[49, 60], [138, 150], [162, 174], [195, 210]],
                 [[63, 80], [97, 113], [120, 134], [165, 180], [184, 205]]]
            for key_idx, key in enumerate(test_keys):
                seq = datasets[key]['features'][...]
                gt = datasets[key]['labels'][...]
                gt = torch.from_numpy(gt)
                label_idx = key_idx
                local_label = local_labels[label_idx]

                seq = torch.from_numpy(seq)
                seq = seq.cuda()
                sub_graphs = None
                for n in range(math.ceil(seq.shape[0] / (args.fragment_length * 2))):
                    if n != math.ceil(seq.shape[0] / (args.fragment_length * 2)) - 1:
                        data = seq[(args.fragment_length * 2) * n:(args.fragment_length * 2) * (n + 1), :]
                        sub_graph, _ = model1(data)
                    else:
                        data = seq[(args.fragment_length * 2) * n:, :]
                        if data.shape[0] == 1:
                            sub_graph = data
                        else:
                            sub_graph, _ = model1(data)

                    if sub_graphs is not None:
                        sub_graphs = torch.cat((sub_graphs, sub_graph), dim=0)
                    else:
                        sub_graphs = sub_graph
                seq_graph = torch.add(seq, sub_graphs)
                seq2seq = seq_graph.unsqueeze(dim=0)
                sig_probs = model2(seq2seq)

                probs_importance = sig_probs.data.cpu().squeeze().numpy()

                seq = seq.squeeze()
                # limits = int(math.floor(seq.shape[0] * prop))
                limits = args.num_fragment
                order = np.argsort(probs_importance)[::-1]

                all_fragment = []
                n_t = 0
                if label_idx != 7 and label_idx != 10:
                    for j in range(len(local_label)):
                        for i in range(limits):
                            gt_left_idx = local_label[j][0]
                            gt_right_idx = local_label[j][1]
                            idx = order[i] + args.fragment_length
                            left_idx = idx - probs_importance[idx - args.fragment_length] * args.fragment_length
                            left_int_idx = int(np.ceil(left_idx))
                            right_idx = idx + probs_importance[idx - args.fragment_length] * args.fragment_length
                            right_int_idx = int(np.floor(right_idx))
                            if left_int_idx - args.fragment_length >= gt_right_idx or right_int_idx - args.fragment_length <= gt_left_idx:
                                tIOU = 0.
                            else:
                                idx_set = np.hstack((gt_left_idx, gt_right_idx))
                                idx_set = np.hstack((idx_set, left_int_idx - args.fragment_length))
                                idx_set = np.hstack((idx_set, right_int_idx - args.fragment_length))
                                idx_set = np.sort(idx_set)
                                tIOU = (idx_set[2] - idx_set[1]) / (idx_set[3] - idx_set[0])
                            if tIOU >= 0.5:
                                n_t += 1
                                break
                    local_recall = n_t / len(local_label)
                    log_str1 = 'i_th trial %.0f\tRecall %.02f' % (
                        label_idx, local_recall)
                    save_idx.write(log_str1 + '\n')
                    save_idx.flush()

                for i in range(limits):
                    idx = order[i] + args.fragment_length
                    left_idx = idx - probs_importance[idx - args.fragment_length] * args.fragment_length
                    left_int_idx = int(np.ceil(left_idx)) - args.fragment_length

                    right_idx = idx + probs_importance[idx - args.fragment_length] * args.fragment_length
                    right_int_idx = int(np.floor(right_idx)) - args.fragment_length
                    one_fragment = seq[left_int_idx:right_int_idx, ]
                    all_fragment.append(one_fragment)

                    log_str0 = 'i_th fragment %.0f\tleft_idx %.0f\tright_idx %.0f' % (
                        i, left_int_idx, right_int_idx)
                    save_idx.write(log_str0 + '\n')
                    save_idx.flush()

                all_fragment = torch.vstack(all_fragment)
                print(all_fragment.shape[0])
                labels = torch.full((all_fragment.shape[0], 1), gt)
                num_segments_trial.append(all_fragment.shape[0])

                if all_features is not None:
                    all_features = torch.cat((all_features, all_fragment), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)
                else:
                    all_features = all_fragment
                    all_labels = labels
            all_features = all_features.cpu().data.numpy()
            all_labels = all_labels.cpu().data.numpy()
            print(all_features.shape, all_labels.shape)
            mat_file = os.path.join(out_path, 'TAS_' + 'subject' + str(args.subject_id)  + '_' + str(args.reward_function) + '_' + str(args.num_fragment) + '.mat')
            savemat(mat_file, {'feature': all_features, 'label': all_labels})
