import argparse
import os
import time

import nflows.nn as nn_
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import distributions
from nflows import flows
from nflows import transforms
from nflows import utils

from utils import arg_util
from utils.path_util import get_absolute_path

PROJECT_PATH = "../"
GPU_ID = 1
dataset_name = "BJAQ"

""" network parameters """
hidden_features = 56
num_flow_steps = 6

flow_id = 1

features = 5

REUSE_FROM_FILE = False
REUSE_FILE_PATH = PROJECT_PATH + "train/"

""" query settings"""
query_seed = 45
QUERY_CNT = 2000

""" detailed network parameters"""
anneal_learning_rate = True
base_transform_type = "rq-coupling"

dropout_probability = 0
grad_norm_clip_value = 5.0
linear_transform_type = "lu"

num_bins = 8
num_training_steps = 400000
num_transform_blocks = 2
seed = 1638128
tail_bound = 3
use_batch_norm = False


def create_linear_transform():
    if linear_transform_type == "permutation":
        return transforms.RandomPermutation(features=features)
    elif linear_transform_type == "lu":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.LULinear(features, identity_init=True),
            ]
        )
    elif linear_transform_type == "svd":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.SVDLinear(features, num_householder=10, identity_init=True),
            ]
        )
    else:
        raise ValueError


def create_base_transform(i):
    # tmp_mask = utils.create_alternating_binary_mask(features, even=(i % 2 == 0))
    return transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_transform_blocks,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        ),
        num_bins=num_bins,
        tails="linear",
        tail_bound=tail_bound,
        apply_unconditional_transform=True,
    )


# torch.masked_select()
def create_transform():
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [create_linear_transform(), create_base_transform(i)]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform()]
    )
    return transform


def load_model(device, model_name: str = "BJAQ"):
    distribution = distributions.StandardNormal((features,))
    transform = create_transform()
    flow = flows.Flow(transform, distribution)

    # if 'Ti' in DEVICENAME:
    #     path = os.path.join(PROJECT_PATH+'train/models/{}'.format(dataset_name),
    #                         '{}-id{}-best-val.t'.format(dataset_name, flow_id))

    # else:
    #     assert False

    # TODO:
    #   1. 若是对于census/power数据集，应使用哪个模型？共有4个可选项(origin/FT/update/adapt)
    #   2. FACE/train/models/power/power-id1-best-val.t报错size mismatch
    path = os.path.join(
        PROJECT_PATH + "train/models/{}".format(model_name),
        "{}-id{}-best-val.t".format(model_name, flow_id),
    )

    print("Load model from:", path)

    flow.load_state_dict(torch.load(path))

    # flow.cuda()
    flow.eval()

    n_params = utils.get_num_parameters(flow)
    # print('There are {} trainable parameters in this model.'.format(n_params))
    # print('Parameters total size is {} MB'.format(n_params * 4 / 1024 / 1024))

    return flow


def normalized(data):
    max = np.max(data, keepdims=True)
    min = np.min(data, keepdims=False)
    _range = max - min
    for i in range(_range.shape[1]):
        if _range[0][i] == 0:
            _range[0][i] = 1
    return (data - min) / _range


def sampling(data, size, replace):
    if not replace and size > data.shape[0]:
        raise ValueError("Size cannot be greater than the number of rows in data when replace is False")

    sample_idx = np.random.choice(range(data.shape[0]), size=size, replace=replace)
    sample_idx = np.sort(sample_idx)
    sample = data[sample_idx]
    return sample


def permute(data, size):
    samples = np.empty(shape=(0, data.shape[1]))
    for _ in range(size):
        idxs = np.random.choice(range(data.shape[0]), data.shape[1], replace=False)
        sample = np.empty(shape=(1, data.shape[1]))
        for i, idx in enumerate(idxs):
            sample[0][i] = data[idx][i]

        samples = np.concatenate((samples, sample), axis=0)

    return samples.astype(np.float32)


def single_sampling(data, size):
    idx = np.random.randint(data.shape[0])
    sample_idx = [idx] * size
    sample = data[sample_idx]
    return sample


def data_update(data, size):
    update_data = sampling(data, size)
    return update_data


def js_div(p_output, q_output, get_softmax=False):
    """
    Function that measures JS divergence between target and output logits:
    """

    KLDivLoss = nn.KLDivLoss(reduction="batchmean")
    p_output = torch.from_numpy(p_output)
    q_output = torch.from_numpy(q_output)
    # print("q_output shape: {}".format(q_output.shape))
    if get_softmax:
        p_output = F.softmax(p_output, dim=0)
        q_output = F.softmax(q_output, dim=0)
    log_mean_output = ((p_output + q_output) / 2).log()
    # print("P_output shape: {}".format(p_output.shape))
    # print("data sample: {}".format(q_output[:5]))
    return (
            KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)
    ) / 2


def loss(sample, flow):
    log_density = flow.log_prob(sample)
    loss = -torch.mean(log_density)
    std = torch.std(log_density)
    return loss.item(), std.item()


def conca_and_save(save_file, old_data, update_data):
    new_data = np.concatenate((old_data, update_data), axis=0)
    print("new data shape: {}".format(new_data.shape))
    np.save(save_file, new_data)


def loss_test(data, update_data, sample_size, flow):
    loss_start_time = time.time()

    old_sample = sampling(data, sample_size, replace=False)
    new_sample = sampling(update_data, sample_size, replace=False)

    old_mean, old_std = loss(old_sample, flow=flow)
    new_mean, _ = loss(new_sample, flow=flow)
    mean_reduction = abs(new_mean - old_mean)
    threshold = 2 * old_std

    loss_end_time = time.time()
    loss_running_time = loss_end_time - loss_start_time
    print("old loss mean - new loss mean: {:.4f}".format(mean_reduction))
    print("2 * std: {:.4f}".format(threshold))
    print("loss test running time: {:.4f}s".format(loss_running_time))
    return mean_reduction, threshold


def JS_test(data, update_data, sample_size, epoch=32):
    # assert update_type in ["sample", "single", "permute"], "Update type error!"
    js_start_time = time.time()
    js_divergence = []
    for i in range(epoch):
        old_sample = sampling(data, sample_size, replace=False)
        new_sample = sampling(update_data, sample_size, replace=True)
        old_sample_norm = normalized(old_sample)
        new_sample_norm = normalized(new_sample)

        JS_diver = js_div(old_sample_norm, new_sample_norm)
        js_divergence.append(JS_diver)
        # print("Epoch {} JS divergence: {:.4f}".format(i + 1, JS_diver))

    js_divergence = np.array(js_divergence).astype(np.float32)
    js_mean = np.mean(js_divergence)

    js_end_time = time.time()
    js_running_time = js_end_time - js_start_time
    print("Mean JS divergence: {:.4f}".format(js_mean))
    print("JS devergence running time: {:.4f}s".format(js_running_time))
    return JS_diver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bjaq", help="Dataset.")
    parser.add_argument(
        "--run",
        help="running type (init, update, default: update)",
        type=str,
        default="update",
    )
    parser.add_argument(
        "--update",
        help="data update type (sample, single, permute, default: sample) ",
        type=str,
        default="sample",
    )
    parser.add_argument(
        "--init_size",
        help="initial data size when run==init, default: 200000",
        type=int,
        default=200000,
    )
    parser.add_argument(
        "--update_size",
        help="update insert size when run==update, default: 10000",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--sample_size",
        help="sample size for update data when run==update, default: 10000",
        type=int,
        default=20000,
    )
    args = parser.parse_args()
    assert args.run in ["init", "update"], "Running Type Error!"
    assert args.update in ["sample", "single", "permute"], "Update Type Error!"
    assert (
            args.update_size >= args.sample_size
    ), "Error! Update Size Must Be Greater Than Sample Size!"
    return args


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

    # 提取参数
    args = parse_args()
    arg_util.validate_argument(arg_util.ArgType.DATASET, args.dataset)
    ini_data_size = args.init_size

    dataset_name = args.dataset
    if dataset_name in ["census", "forest", "bjaq", "power"]:
        raw_file_path = f"./data/{dataset_name}/{dataset_name}.npy"
        sampled_file_path = f"./data/{dataset_name}/sampled/{dataset_name}-sample{ini_data_size}.npy"
    else:
        return
    raw_file_path = get_absolute_path(raw_file_path)
    sampled_file_path = get_absolute_path(sampled_file_path)

    model_name = "BJAQ" if dataset_name == "bjaq" else dataset_name
    flow = load_model(device, model_name=model_name)

    # 为原始数据集创建子集
    if args.run == "init":
        raw_data = np.load(raw_file_path, allow_pickle=True)
        ini_data = sampling(raw_data, ini_data_size, replace=False)
        print(ini_data.shape)
        os.makedirs(os.path.dirname(sampled_file_path), exist_ok=True)
        np.save(sampled_file_path, ini_data)
        print(sampled_file_path, "saved")

    # 抽取增量更新数据，更新数据集，并进行数据漂移判定，输出mean reduction、2*std、Mean JS divergence三个参数
    if args.run == "update":
        update_size = args.update_size
        sample_size = args.sample_size
        data = np.load(sampled_file_path).astype(np.float32)
        # data=np.load(root_file).astype(np.float32)

        if args.update == "permute":
            update_data = permute(data, update_size)
        elif args.update == "sample":
            update_data = sampling(data, update_size, replace=True)
        else:
            update_data = single_sampling(data, update_size)

        # update_data = np.concatenate((data, update_data), axis=0)

        mean_reduction, threshold = loss_test(data, update_data, sample_size, flow=flow)
        # print("sample dtype: {}".format(old_sample.dtype))
        JS_diver = JS_test(data, update_data, sample_size)
        conca_and_save(sampled_file_path, data, update_data)

    # # print("data sample: {}".format(input[:5]))


if __name__ == "__main__":
    main()
