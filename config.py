params = {
    # Train
    "epochs": 20000,
    "learning_rate": 1e-4,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,
    "width":0,
    # Early-stopping
    "no_improve_epochs": 80,

    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,

    # Dataset
    "target": "ucddb",
    "dataset":["ucddb","sleepedf","isruc"],
    "kfold":5,
    "use_adv":True,
    "cross_domain":True,
    "gpu":1,
    "suffix":"no_noisy",
    # Data Augmentation
    "weighted_cross_ent": True,
    "augment_seq":False,

    #da
    "num_steps":300,
    "eval_step":300
    #set
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 16,
})

predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 1,
})
