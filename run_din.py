# coding=utf-8
import sys

sys.path.insert(0, '..')

import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN


def get_xy_fd():
    # ——*——*——*——*——*——*——*——*——*—— 特征嵌入 ——*——*——*——*——*——*——*——*——*——
    feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                       DenseFeat('score', 1)]  # 类别特征嵌入

    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4, length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4,
                                         length_name="seq_length")]  # 可变长的类型特征——历史序列数据
    # ——*——*——*——*——*——*——*——*——*—— 初始化虚拟数据 ——*——*——*——*——*——*——*——*——*——
    behavior_feature_list = ["item", "item_gender"]  # 可变长的类别特征列表
    uid = np.array([0, 1, 2])  # 用户id
    ugender = np.array([0, 1, 0])  # 用户性别特征
    iid = np.array([1, 2, 3])  # 0 is mask value 被推荐的商品id
    igender = np.array([1, 2, 1])  # 0 is mask value  被推荐的商品类别
    score = np.array([0.1, 0.2, 0.3])  # 用户的评分特征

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])  # 用户历史行为序列，不足补0
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    behavior_length = np.array([3, 3, 2])  # 用户历史行为序列长度

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
                    "seq_length": behavior_length}  # 构造特征字典

    # 构造输入（x，y）
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])  # 1: 正样本, 0: 负样本

    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():  # pytorch-gpu
        print('cuda ready...')
        device = 'cuda:0'

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, batch_size=3, epochs=10, verbose=2, validation_split=0.0)
