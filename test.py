import numpy as np
import sklearn.metrics as skmetrics
import torch
from minibatching import (iterate_batch_multiple_seq_minibatches)
from util import (getdata)


def predict(
    config,target
):

    target_set=[target]
    print(target_set)
    acc_test_list = []
    mf1_test_list = []

    #kfold
    kfold=5
    for fold in range(kfold):
        for ds in target_set:
            file_path = './data/{}'.format(ds)
            files = os.listdir(file_path)
            test_sids_npz = np.load('./k_fold/{}.npz'.format(ds))
            test_sids = test_sids_npz['k_fold{}'.format(fold)]
            test_x, test_y = getdata(file_path, test_sids)
        # load model
        print('./model/{}/kold{}_best.pth'.format(target,fold))
        model = torch.load('./model/{}/kold{}_best.pth'.format(target,fold))

        t_preds, t_trues = [], []
        test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            test_x,
            test_y,
            0,
            batch_size=config["batch_size"] // 2,
            seq_length=config["seq_length"],
            shuffle_idx=None,
            augment_seq=False,
        )
        state_test = []
        for j in range(3):
            state_test.append((torch.zeros(size=(1, 8, 128)).cuda(),
                               torch.zeros(size=(1, 8, 128)).cuda()))
        for x, y, d, w, sl, re, _ in test_minibatch_fn:
            x = torch.tensor(x, dtype=torch.float32).cuda()
            y = torch.tensor(y, dtype=torch.long).cuda()
            if re:
                for j in range(3):
                    state_test[j] = ((torch.zeros(size=(1, 8, 128)).cuda(),
                                      torch.zeros(size=(1, 8, 128)).cuda()))
            with torch.no_grad():
                t_pred0, t_pred1, t_pred4, state_test, t_export0, t_export1,w = model.valid(data_tgt=x,
                                                                                          state_t=state_test)

                t_pred = t_export0 + t_export1
                t_pred = torch.max(t_pred, 1)[1]

            tmp_preds = np.reshape(t_pred.cpu(), (config["batch_size"] // 2, config["seq_length"]))
            tmp_trues = np.reshape(y.cpu(), (config["batch_size"] // 2, config["seq_length"]))

            for i in range(config["batch_size"] // 2):
                t_preds.extend(tmp_preds[i, :sl[i]])
                t_trues.extend(tmp_trues[i, :sl[i]])

        t_acc = skmetrics.accuracy_score(y_true=t_trues, y_pred=t_preds)
        t_f1_score = skmetrics.f1_score(y_true=t_trues, y_pred=t_preds, average="macro")


        acc_test_list.append(t_acc)
        mf1_test_list.append(t_f1_score)
    avg_test_acc = 0
    avg_test_mf1 = 0
    for i in range(kfold):
        avg_test_acc = avg_test_acc + acc_test_list[i]
        avg_test_mf1 = avg_test_mf1 + mf1_test_list[i]
        print("kfold{} :test acc ={:.3f},f1={:.3f}".format(i, acc_test_list[i],mf1_test_list[i]))

    avg_test_acc = avg_test_acc / kfold
    avg_test_mf1 = avg_test_mf1 / kfold

    print("average :test acc ={:.3f},f1={:.3f}".format(avg_test_acc, avg_test_mf1))

    return 0

if __name__ == "__main__":
    import argparse
    import importlib
    import os
    config_file = os.path.join('config.py')
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.train
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='sleepedf', help='target dataset')
    args = parser.parse_args()
    predict(config,args.target)