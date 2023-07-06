import os
import time
import torch
import argparse
from tqdm import tqdm
import pickle
from transformers import AutoModel, AutoConfig
from torch.utils.data import DataLoader

from cv_datasets import CVDataset
from utils import fix_random_seed

BASE_DIR = "/home/brian/work/ICLR-Representation_Transferability_X/cv_emd"
_PRETRAIN_CONFIG_HUB = {
    'vit-mnist': ["farleyknight-org-username/vit-base-mnist", # Imagenet 21k
                    os.path.join(BASE_DIR, "finetuned_vit_model/farleyknight-org-username__vit-base-mnist")],
    'vit-fmnist': ["abhishek/autotrain_fashion_mnist_vit_base", # Imagenet 1k
                    os.path.join(BASE_DIR, "finetuned_vit_model/abhishek__autotrain_fashion_mnist_vit_base")],
    'vit-cifar10': ["aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", # Imagenet 21k
                    os.path.join(BASE_DIR, "finetuned_vit_model/aaraki__vit-base-patch16-224-in21k-finetuned-cifar10")],
}
_DATA_HUB = {
    'vit-mnist': os.path.join(BASE_DIR, "cv_datasets/mnist"),
    'vit-fmnist': os.path.join(BASE_DIR, "cv_datasets/fashion_mnist"),
    'vit-cifar10': os.path.join(BASE_DIR, "cv_datasets/cifar10")
}

def main(args):
    fix_random_seed(args.seed)
    ts = time.time()
    # Prepare data
    set_name = _DATA_HUB[args.model].split('/')[-1]
    print("Loading dataset {}...".format(set_name.upper()))
    dataset = CVDataset(_DATA_HUB[args.model], set_name, 'test', return_labels=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    # Construct VIT model with huggingface
    model_config = AutoConfig.from_pretrained(_PRETRAIN_CONFIG_HUB[args.model][0])
    if os.path.exists(_PRETRAIN_CONFIG_HUB[args.model][1]): # load locally
        print("Building VIT model locally...")
        VIT_model = AutoModel.from_pretrained(_PRETRAIN_CONFIG_HUB[args.model][1], config=model_config)
    else:
        print("Building VIT model online...")
        VIT_model = AutoModel.from_pretrained(_PRETRAIN_CONFIG_HUB[args.model][0], config=model_config)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    VIT_model.to(device)

    # Start one-epoch forward to obtain feature embeddings
    embeddings, labels = [], []
    VIT_model.eval()
    for  batch in tqdm(dataloader, desc='{} forward'.format(set_name.upper())):
        img_inputs, img_labels = batch
        img_inputs = img_inputs.to(device)
        with torch.no_grad():
            outputs = VIT_model(img_inputs)
            last_hidden_states = outputs.last_hidden_state.detach().cpu()
            last_hidden_states_mean = last_hidden_states[:, 1:, :].mean(dim=1) # w/o CLS token
            embeddings.append(last_hidden_states_mean)
            labels.append(img_labels)
    embeddings = torch.concat(embeddings, dim=0).numpy()
    labels = torch.concat(labels, dim=0).numpy()

    # Save embeddings and labels
    emd_and_lbl = {
        'embeddings': embeddings,
        'labels': labels
    }
    emd_lbl_dir = os.path.join(BASE_DIR, 'emd_lbl')
    os.makedirs(emd_lbl_dir, exist_ok=True)
    with open(os.path.join(emd_lbl_dir, f'{set_name}_emd_lbl.pkl'), 'wb') as handle:
        pickle.dump(emd_and_lbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finish inference at {:.1f} mins. Embeddings and labels are stored at `{}`".format(
            (time.time()-ts)/60, emd_lbl_dir))


def args_parser():
    parser = argparse.ArgumentParser("VIT Embedder")
    parser.add_argument("--model", type=str, default='vit-mnist', choices=_PRETRAIN_CONFIG_HUB.keys())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1145114)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    main(args)
