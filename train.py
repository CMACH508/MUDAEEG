import copy
import os
import numpy as np
import sklearn.metrics as skmetrics
import shutil
import torch
from tqdm import tqdm
from model_target import TinySleepNet
from minibatching import (iterate_batch_multiple_seq_minibatches_loop,
                          iterate_batch_multiple_seq_minibatches)
from util import getdata
from logger import get_logger

def train_torch(
    config,
    output_dir,
    log_file,
    restart=False,
):

    output_dir = os.path.join(output_dir,"out")
    dataset = config["dataset"]
    target = config["target"]
    target_set=[target]
    print(target_set)
    source_set = [i for i in dataset if i not in target_set]
    print(source_set)


    acc_valid_list = []
    mf1_valid_list = []
    acc_test_list = []
    mf1_test_list = []
    # Create output directory for the specified fold_idx
    if restart:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # Create logger
    logger = get_logger(log_file, level="info")
    logger.info("source_set :{}".format(source_set))
    source_num = len(source_set)
    config["source_num"]=source_num
    train_xs=[[] for i in range(source_num)]
    train_ys=[[] for i in range(source_num)]
    # Load subject IDs

    for i in range(source_num):
        ds = source_set[i]
        print(ds)
        file_path = './data/{}'.format(ds)
        files = os.listdir(file_path)
        train_xs[i], train_ys[i] = getdata(file_path, files)
        print(len(train_xs[i]), len(train_ys[i]), train_xs[i][0].shape,len(train_ys[i][0]))

    #kfold
    kfold = config["kfold"]
    for fold in range(kfold):
        logger.info("k_fold:{}".format(fold))
        source_minibatches = [[] for i in range(source_num)]
        if (config["use_adv"]):
            for i in range(source_num):
                source_minibatches[i] = iterate_batch_multiple_seq_minibatches_loop(
                    train_xs[i],
                    train_ys[i],
                    i,
                    batch_size=config["batch_size"] // 2,
                    seq_length=config["seq_length"],
                    shuffle_idx=None,
                    augment_seq=config['augment_seq'],
                )


        for ds in target_set:
            file_path = './data/{}'.format(ds)
            files = os.listdir(file_path)
            test_sids_npz = np.load('./k_fold/{}.npz'.format(ds))
            test_sids = test_sids_npz['k_fold{}'.format(fold)]
            train_t_sids = np.setdiff1d(files, test_sids)

            n_valid = round(len(train_t_sids) * 0.25)

            if target == 'sleepedf':
                valid_sids = []
                while len(valid_sids) < n_valid:
                    tag = np.random.choice(20)
                    if tag < 10:
                        name1 = 'SC40{}1E0.npz'.format(tag)
                        name2 = 'SC40{}2E0.npz'.format(tag)
                        if name1 in train_t_sids and name2 in train_t_sids:
                            valid_sids.append(name1)
                            valid_sids.append(name2)
                    elif tag == 13:
                        name = 'SC4131E0.npz'
                        if name in train_t_sids:
                            valid_sids.append(name)
                    else:
                        name1 = 'SC4{}1E0.npz'.format(tag)
                        name2 = 'SC4{}2E0.npz'.format(tag)
                        if name1 in train_t_sids and name2 in train_t_sids:
                            valid_sids.append(name1)
                            valid_sids.append(name2)
                    train_t_sids = np.setdiff1d(train_t_sids, valid_sids)

            else:
                valid_sids = np.random.choice(train_t_sids, n_valid, replace=False)
                train_t_sids = np.setdiff1d(train_t_sids, valid_sids)

            logger.info("Train SIDs: ({}) {}".format(len(train_t_sids), train_t_sids))
            logger.info("Valid SIDs: ({}) {}".format(len(valid_sids), valid_sids))
            logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))
            train_xt, train_yt = getdata(file_path, train_t_sids)
            valid_x, valid_y = getdata(file_path, valid_sids)
            test_x, test_y = getdata(file_path, test_sids)
            logger.info("Training set (n_night_sleeps={})".format(len(train_yt)))
            logger.info("valid set (n_night_sleeps={})".format(len(valid_y)))
            logger.info("Test set (n_night_sleeps={})".format(len(test_y)))


        batch_size_t = config["batch_size"]
        target_minibatch = iterate_batch_multiple_seq_minibatches_loop(
            train_xt,
            train_yt,
            source_num,
            batch_size=batch_size_t // 2,
            seq_length=config["seq_length"],
            shuffle_idx=None,
            augment_seq=config['augment_seq'])

        # Create a model
        model = TinySleepNet(config=config).cuda()

        best_valid_acc = -1
        best_valid_mf1 = -1
        best_test_acc = -1
        best_test_mf1 = -1

        states = []
        state_s,state_t = [],[]
        for i in range(2):
            states.append((torch.zeros(size=(1, 8, 128)).cuda(),
                 torch.zeros(size=(1, 8, 128)).cuda()))
        for i in range(2):
            state_s.append(copy.deepcopy(states))
            state_t.append(copy.deepcopy(states))
        epochs = config["epochs"]
        for step in tqdm(range(0, epochs)):
            # Create minibatches for training
            for dom in range(source_num):
                xs, ys, ds, ws, sls, res, _ = next(source_minibatches[dom])
                if dom%source_num==0:
                    xt, yt, dt, wt, slt, ret, _ = next(
                        target_minibatch)
                    xt = torch.tensor(xt, dtype=torch.float32).cuda()
                    wt = torch.from_numpy(wt).cuda()
                xs = torch.tensor(xs, dtype=torch.float32).cuda()
                ys = torch.tensor(ys, dtype=torch.long).cuda()
                ws = torch.from_numpy(ws).cuda()
                if res:
                    state_s[dom] = copy.deepcopy(states)
                if ret:
                    state_t[dom] = copy.deepcopy(states)

                cls_loss_mix, cls_loss,loss_dis,loss_export, state_s,state_t = model.train_da(data_src=xs, data_tgt=xt, label_src=ys, mark=dom, state_s=state_s, state_t=state_t,sw=ws,tw=wt)


            if (step % 300 == 0):
                logger.info("step={}".format(step))
            if step % 300 == 0:

                v_trues,v_exports = [], []

                valid_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                    test_x,
                    test_y,
                    source_num,
                    batch_size=config["batch_size"] // 2,
                    seq_length=config["seq_length"],
                    shuffle_idx=None,
                    augment_seq=False,
                )
                state_valid = []
                for j in range(3):
                    state_valid.append((torch.zeros(size=(1, 8, 128)).cuda(),
                                       torch.zeros(size=(1, 8, 128)).cuda()))
                for x, y, d, w, sl, re, _ in valid_minibatch_fn:
                    x = torch.tensor(x, dtype=torch.float32).cuda()
                    y = torch.tensor(y, dtype=torch.long).cuda()
                    if re:
                        for j in range(3):
                            state_valid[j] = ((torch.zeros(size=(1, 8, 128)).cuda(),
                                              torch.zeros(size=(1, 8, 128)).cuda()))
                    with torch.no_grad():
                        v_pred0, v_pred1, v_pred4, state_valid,v_export0,v_export1,we = model.valid(data_tgt=x, state_t=state_valid)  # t_pred4是域不变分支结果，即所有源域数据都要经过它
                        v_export = v_export1
                        v_export = torch.max(v_export, 1)[1]

                    tmp_preds_export = np.reshape(v_export.cpu(), (config["batch_size"] // 2, config["seq_length"]))
                    tmp_trues = np.reshape(y.cpu(), (config["batch_size"] // 2, config["seq_length"]))

                    for i in range(config["batch_size"] // 2):
                        v_exports.extend(tmp_preds_export[i, :sl[i]])
                        v_trues.extend(tmp_trues[i, :sl[i]])


                v_acc_export = skmetrics.accuracy_score(y_true=v_trues, y_pred=v_exports)
                v_f1_score_export = skmetrics.f1_score(y_true=v_trues, y_pred=v_exports, average="macro")

                logger.info("valid export :acc ={:.3f}, export f1={:.3f}".format(v_acc_export, v_f1_score_export))

                if best_valid_acc < v_acc_export :
                    best_valid_acc = v_acc_export
                    best_valid_mf1 = v_f1_score_export

                    torch.save(model, './model/{}/'.format(config['target']) + 'kold{}_best.pth'.format(fold))

                    t_preds, t_trues = [], []
                    test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                        test_x,
                        test_y,
                        source_num,
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
                            t_pred0, t_pred1, t_pred4, state_test,t_export0,t_export1,we = model.valid(data_tgt=x, state_t=state_test)
                            t_pred = t_export0+t_export1
                            t_pred = torch.max(t_pred, 1)[1]

                        tmp_preds = np.reshape(t_pred.cpu(), (config["batch_size"] // 2, config["seq_length"]))
                        tmp_trues = np.reshape(y.cpu(), (config["batch_size"] // 2, config["seq_length"]))

                        for i in range(config["batch_size"] // 2):
                            t_preds.extend(tmp_preds[i, :sl[i]])
                            t_trues.extend(tmp_trues[i, :sl[i]])

                    t_acc = skmetrics.accuracy_score(y_true=t_trues, y_pred=t_preds)
                    t_f1_score = skmetrics.f1_score(y_true=t_trues, y_pred=t_preds, average="macro")

                    logger.info("test :acc ={:.3f},f1={:.3f}".format(t_acc, t_f1_score))

                    best_test_acc = t_acc
                    best_test_mf1 = t_f1_score

        acc_valid_list.append(best_valid_acc)
        mf1_valid_list.append(best_valid_mf1)
        acc_test_list.append(best_test_acc)
        mf1_test_list.append(best_test_mf1)
    avg_valid_acc = 0
    avg_valid_mf1 = 0
    avg_test_acc = 0
    avg_test_mf1 = 0
    for i in range(kfold):
        avg_valid_acc = avg_valid_acc + acc_valid_list[i]
        avg_valid_mf1 = avg_valid_mf1 + mf1_valid_list[i]
        avg_test_acc = avg_test_acc + acc_test_list[i]
        avg_test_mf1 = avg_test_mf1 + mf1_test_list[i]
        logger.info("kfold{} :valid acc ={:.3f},f1={:.3f},test acc ={:.3f},f1={:.3f}".format(i, acc_valid_list[i],
                                                                                                 mf1_valid_list[i],
                                                                                                 acc_test_list[i],
                                                                                                 mf1_test_list[i]))

    avg_valid_acc = avg_valid_acc / kfold
    avg_valid_mf1 = avg_valid_mf1 / kfold
    avg_test_acc = avg_test_acc / kfold
    avg_test_mf1 = avg_test_mf1 / kfold

    logger.info(
            "average :valid acc ={:.3f},f1={:.3f},test acc ={:.3f},f1={:.3f}".format(avg_valid_acc, avg_valid_mf1,
                                                                                     avg_test_acc, avg_test_mf1))

    return 0