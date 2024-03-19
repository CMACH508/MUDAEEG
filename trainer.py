import argparse
import importlib
import os
from train import train_torch



def run(config,target,gpu):
    config["target"] = target
    config["gpu"] = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])

    # Output directory
    output_dir = f'_{config["target"]}_{config["cross_domain"]}_{config["use_adv"]}_{config["suffix"]}'#_shhs2_True_True_no_noisy
    print(config)

    # Training
    v = train_torch(
        config=config,
        output_dir=os.path.join(output_dir, 'train'),
        log_file=os.path.join(output_dir, f'train_{config["gpu"]}.log'),
        restart = True,
    )


if __name__ == '__main__':
    config_file = os.path.join('config.py')
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.train
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='ucddb', help='target dataset') #target dataset
    parser.add_argument('--gpu', type=int, default=1, help='gpu') #gpu
    args = parser.parse_args()
    run(config,args.target,args.gpu)