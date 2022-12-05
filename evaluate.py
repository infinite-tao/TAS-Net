import torch
import h5py
import os
from tabulate import tabulate
import numpy as np
import random
import math
from scipy.io import savemat
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(args, model1, model2, dataset, test_keys):

    with torch.no_grad():
        model1.eval()
        model2.eval()
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
        local_recalls = []
        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            gt = dataset[key]['labels'][...]
            gt = torch.from_numpy(gt)
            label_idx = key_idx
            local_label = local_labels[label_idx]
            seq = torch.from_numpy(seq)  # input shape (seq_len, dim)
            seq = seq.to(DEVICE)

            local_graphs0 = None
            for n in range(math.ceil(seq.shape[0] / (args.fragment_length * 2))):
                if n != math.ceil(seq.shape[0] / (args.fragment_length * 2)) - 1:
                    data0 = seq[(args.fragment_length * 2) * n:(args.fragment_length * 2) * (n + 1), :]
                    sub_graph0, _ = model1(data0)
                else:
                    data0 = seq[(args.fragment_length * 2) * n:, :]
                    if data0.shape[0] == 1:
                        sub_graph0 = data0
                    else:
                        sub_graph0, _ = model1(data0)

                if local_graphs0 is not None:
                    local_graphs0 = torch.cat((local_graphs0, sub_graph0), dim=0)
                else:
                    local_graphs0 = sub_graph0
            #
            seq_graph0 = torch.add(seq, local_graphs0)
            seq_graph0 = seq_graph0.unsqueeze(dim=0)

            sig_probs = model2(seq_graph0)

            probs_importance = sig_probs.data.cpu().squeeze().numpy()

            limits = args.num_fragment
            order = np.argsort(probs_importance)[::-1]

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
                local_recalls.append(local_recall)
    return np.mean(local_recalls, axis=0)