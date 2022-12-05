import argparse
import torch
import numpy as np
import os
import scipy.io as scio
import torch.utils.data as Data
import copy
from Model_architecture import EEGfuseNet_Channel_32, EEGfuseNet_Channel_62
from sklearn import preprocessing
import utils
import h5py

def data_norm(X_data,resolution):
    X_data_sque=np.reshape(X_data,(np.shape(X_data)[0],32*resolution))
    X_data_sque_scaled=max_min_scale(X_data_sque)
    X_data=np.reshape(X_data_sque_scaled,(np.shape(X_data)[0],32,resolution))
    return X_data
def max_min_scale(data_sque):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    data_sque=min_max_scaler.fit_transform(data_sque)
    return data_sque

def get_dataset(norm_type, resolution):
    data_list = []
    for j in range(32):
        sub = 'sub_' + str(j + 1)
        # os.chdir('F:\\zhourushuang\\dataset_auto_384')
        data_path = r'/home/ubuntu/zhangyongtao/PycharmProjects/PHDwork/Datasets/dataset_auto_384/'
        path = os.path.join(data_path, sub)
        times = 1
        for info in os.listdir(path):
            domain = os.path.abspath(path)  # 获取文件夹的路径
            info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
            data = scio.loadmat(info)['Data'][0, 0]['eeg_data']
            for i in range(times):
                data_list.append(data)

    X_data = np.reshape(data_list, (np.shape(data_list)[0], 1, 32, 384))
    if norm_type == 'ele':
        data_tmp = copy.deepcopy(X_data)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    if norm_type == 'global_scale_value':
        data_tmp = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
    if norm_type == "global_gaussian_value":
        data_tmp = (X_data - np.mean(X_data)) / np.std(X_data)
    if norm_type == "pixel_scale_value":
        data_tmp = data_norm(X_data, resolution)
    if norm_type == "origin":
        data_tmp = X_data
        # X_data, valid_data = train_test_split(X_data, test_size=0.1, random_state=0)
    return data_tmp


def get_dataset_hci(norm_type, resolution):
    data_list = []
    for j in range(24):
        sub = 'sub_' + str(j + 1)
        data_path = r'/home/ubuntu/zhangyongtao/PycharmProjects/PHDwork/Datasets/dataset_auto_384_hci/'
        path = os.path.join(data_path, sub)
        times = 1
        for info in os.listdir(path):
            domain = os.path.abspath(path)  # 获取文件夹的路径
            info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
            data = scio.loadmat(info)['Data'][0, 0]['eeg_data']
            for i in range(times):
                data_list.append(data)

    X_data = np.reshape(data_list, (np.shape(data_list)[0], 1, 32, 384))
    if norm_type == 'ele':
        data_tmp = copy.deepcopy(X_data)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    if norm_type == 'global_scale_value':
        data_tmp = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
    if norm_type == "global_gaussian_value":
        data_tmp = (X_data - np.mean(X_data)) / np.std(X_data)
    if norm_type == "pixel_scale_value":
        data_tmp = data_norm(X_data, resolution)
    if norm_type == "origin":
        data_tmp = X_data
        # X_data, valid_data = train_test_split(X_data, test_size=0.1, random_state=0)
    return data_tmp


def get_dataset_seed(norm_type, resolution, Session):
    data_list = []
    label_list = []
    for j in range(15):
        for s in range(3):
            if s == Session:
                sub = 'sub_' + str(j + 1)
                session = 'session' + str(s + 1)
                data_path = r'/home/ubuntu/zhangyongtao/PycharmProjects/PHDwork/Datasets/dataset_auto_384_seed/'
                path_im = os.path.join(data_path, sub)
                path = os.path.join(path_im, session)
                times = 1
                for info in os.listdir(path):
                    domain = os.path.abspath(path)  # 获取文件夹的路径
                    info = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
                    data, label = scio.loadmat(info)['Data'][0, 0]['eeg_data'], scio.loadmat(info)['Data'][0, 0][
                        'label']
                    for i in range(times):
                        data_list.append(data)
                        label_list.append(label)
    X_data = np.reshape(data_list, (np.shape(data_list)[0], 1, 62, 384))
    X_label = np.reshape(label_list, (np.shape(data_list)[0], 1))
    if norm_type =='ele':
        data_tmp = copy.deepcopy(X_data)
        label_tmp = copy.deepcopy(X_label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    if norm_type == 'global_scale_value':
        data_tmp = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
    if norm_type == "global_gaussian_value":
        data_tmp = (X_data - np.mean(X_data)) / np.std(X_data)
    if norm_type == "pixel_scale_value":
        data_tmp = data_norm(X_data, resolution)
    if norm_type == "origin":
        data_tmp = X_data
    # X_data, valid_data = train_test_split(X_data, test_size=0.1, random_state=0)
    return data_tmp, label_tmp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEGFUSENet parameters')
    parser.add_argument('--dataset', type=str, default='seed',
                        help='the dataset used for Extract Features, "seed" or "hci" or "deap"')
    parser.add_argument('--norm_type', type=str, default='ele',
                        help='the normalization type used for data, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for one batch, integer')
    parser.add_argument('--session_id', type=int, default=1,
                        help='different session')
    args = parser.parse_args()
    dataset_name = args.dataset
    norm_type = args.norm_type
    BATCH_SIZE = args.batch_size
    Session_ID = args.session_id
    # data preparation
    print('Model name: EEGFuseNet. Dataset name: ', dataset_name)
    print('Normalization type: ', norm_type)
    if dataset_name == 'seed':
        X_data, label_data = get_dataset_seed(norm_type, 384, 1)
        model = EEGfuseNet_Channel_62(16, 1, 1, 384).to(device)  # Hyperparameter
        model.load_state_dict(torch.load('./Pretrained_model_SEED.pkl'))
    if dataset_name == 'hci':
        X_data, label_data = get_dataset_hci(norm_type, 384)
        model = EEGfuseNet_Channel_32(16, 1, 1, 384).to(device)  # Hyperparameter
        model.load_state_dict(torch.load('./Pretrained_model_HCI.pkl'))
    if dataset_name == 'deap':
        X_data, label_data = get_dataset(norm_type, 384)
        model = EEGfuseNet_Channel_32(16, 1, 1, 384).to(device)  # Hyperparameter
        model.load_state_dict(torch.load('./Pretrained_model_DEAP.pkl'))

    X_data = X_data.astype('float32')
    X_data = torch.from_numpy(X_data)
    labels = torch.LongTensor(label_data)
    torch_dataset_train = Data.TensorDataset(X_data, labels)

    loader_data = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    save_path = '/home/ubuntu/zhangyongtao/PycharmProjects/EEGfusenet/features/'
    save_path0 = os.path.join(save_path, 'session_' + str(Session_ID))
    if not os.path.exists(save_path0):
        os.makedirs(save_path0)
    source_h5_file = h5py.File(os.path.join(save_path0, 'source_h5_file.h5'), 'w')
    for i in range(15):
        for j in range(15):
            source_h5_file.create_group('source_{}_video_{}'.format((i + 1), str(j+1).rjust(2, '0')))

    model.eval()
    with torch.no_grad():
        Deep_features = []
        for step, (batch_x, batch_y) in enumerate(loader_data):
            # wrap them in Variable
            inputs = batch_x.to(device)
            # zero the parameter gradients
            _, features = model(inputs)
            Deep_features.append(features)
        all_features = torch.vstack(Deep_features).cpu().data.numpy()

        for i in range(15):
            source_i_data = all_features[3394*i:3394*(i+1), ]
            source_i_label = np.array([2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0])
            source_h5_file['source_{}_video_01'.format(i + 1)]['features'] = list(source_i_data[:235, ])
            source_h5_file['source_{}_video_01'.format(i + 1)]['labels'] = int(source_i_label[0])

            source_h5_file['source_{}_video_02'.format(i + 1)]['features'] = list(source_i_data[235:468, ])
            source_h5_file['source_{}_video_02'.format(i + 1)]['labels'] = int(source_i_label[1])

            source_h5_file['source_{}_video_03'.format(i + 1)]['features'] = list(source_i_data[468:674, ])
            source_h5_file['source_{}_video_03'.format(i + 1)]['labels'] = int(source_i_label[2])

            source_h5_file['source_{}_video_04'.format(i + 1)]['features'] = list(source_i_data[674:912, ])
            source_h5_file['source_{}_video_04'.format(i + 1)]['labels'] = int(source_i_label[3])

            source_h5_file['source_{}_video_05'.format(i + 1)]['features'] = list(source_i_data[912:1097, ])
            source_h5_file['source_{}_video_05'.format(i + 1)]['labels'] = int(source_i_label[4])

            source_h5_file['source_{}_video_06'.format(i + 1)]['features'] = list(source_i_data[1097:1292, ])
            source_h5_file['source_{}_video_06'.format(i + 1)]['labels'] = int(source_i_label[5])

            source_h5_file['source_{}_video_07'.format(i + 1)]['features'] = list(source_i_data[1292:1529, ])
            source_h5_file['source_{}_video_07'.format(i + 1)]['labels'] = int(source_i_label[6])

            source_h5_file['source_{}_video_08'.format(i + 1)]['features'] = list(source_i_data[1529:1745, ])
            source_h5_file['source_{}_video_08'.format(i + 1)]['labels'] = int(source_i_label[7])

            source_h5_file['source_{}_video_09'.format(i + 1)]['features'] = list(source_i_data[1745:2010, ])
            source_h5_file['source_{}_video_09'.format(i + 1)]['labels'] = int(source_i_label[8])

            source_h5_file['source_{}_video_{}'.format((i + 1), 10)]['features'] = list(source_i_data[2010:2247, ])
            source_h5_file['source_{}_video_{}'.format((i + 1), 10)]['labels'] = int(source_i_label[9])

            source_h5_file['source_{}_video_{}'.format((i + 1), 11)]['features'] = list(source_i_data[2247:2482, ])
            source_h5_file['source_{}_video_{}'.format((i + 1), 11)]['labels'] = int(source_i_label[10])

            source_h5_file['source_{}_video_{}'.format((i + 1), 12)]['features'] = list(source_i_data[2482:2715, ])
            source_h5_file['source_{}_video_{}'.format((i + 1), 12)]['labels'] = int(source_i_label[11])

            source_h5_file['source_{}_video_{}'.format((i + 1), 13)]['features'] = list(source_i_data[2715:2950, ])
            source_h5_file['source_{}_video_{}'.format((i + 1), 13)]['labels'] = int(source_i_label[12])

            source_h5_file['source_{}_video_{}'.format((i + 1), 14)]['features'] = list(source_i_data[2950:3188, ])
            source_h5_file['source_{}_video_{}'.format((i + 1), 14)]['labels'] = int(source_i_label[13])

            source_h5_file['source_{}_video_{}'.format((i + 1), 15)]['features'] = list(source_i_data[3188:, ])
            source_h5_file['source_{}_video_{}'.format((i + 1), 15)]['labels'] = int(source_i_label[14])

        print(all_features.shape)

