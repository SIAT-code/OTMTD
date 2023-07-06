import os
import time
import datetime
import random
import yaml
from unittest import loader
from matplotlib.cbook import flatten
import torch
import torch_geometric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import *
from data_transform import load_tensor_for_pro_ids, transform_data, load_tensor, split_dataset, shuffle_dataset, CATHDataset, ProteinGraphDataset
from train_test import pro_mask_tokens, goterm_mask_tokens, remove_no_loss_calculation
import torch.nn as nn


# 固定随机种子以复现结果
SEED = 1
def seed_everything(seed):
    torch.manual_seed(seed)                   # Current CPU
    torch.cuda.manual_seed(seed)              # Current GPU
    np.random.seed(seed)                      # Numpy module
    random.seed(seed)                         # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed)          # All GPU (Optional)

def stable_wrapper(dataloader, seed):
    """ Seed everyting every time after the initialization of torch.data.DataLoader. """
    seed_everything(seed)
    return dataloader


def generate_dataset(args, tokens_input, data_type, task_name, slicer=None, shuffle=False, num_bins=100):
    """ 生成多模态输入数据 """
    print("Generating multimodal input data for task-{}...".format(task_name))
    ts = time.time()
    # 数据兼容性加载：注意，暂时未兼容STRINGx
    if task_name == 'skempi' or task_name == 'kinase':
        df_tokens = pd.read_csv(tokens_input, sep=',') # 带header
        seq_tokens = df_tokens.iloc[:, 2].values.tolist() # seq在第2列
        go_tokens = df_tokens.iloc[:, 3].values.tolist() # go在第3列 df_tokens.iloc[:, 3].values.tolist()
    else: # 'stability', 'fluorescence', 'remote_homology', 'secondary_structure', 'pdbbind'
        df_tokens = pd.read_csv(tokens_input, sep=',', header=None) # 无header
        seq_tokens = df_tokens[1].values.tolist() # seq在第1列
        if task_name == 'pdbbind':
            go_tokens = df_tokens[6].values.tolist() # go在最后一列(第6列)
        else:
            go_tokens = df_tokens[2].values.tolist() # go在第2列
    
    # 随机生成的labels
    # labels = [1 for i in range(len(df_tokens))]
    if task_name == 'pdbbind':
        labels = df_tokens.iloc[:, 4]
    else:
        labels = df_tokens.iloc[:, df_tokens.shape[1]-1]

    seq_tokens_tmp = []
    for seq_token in seq_tokens:
        seq_tokens_tmp.append(seq_token[: args.max_pro_seq_len])
    seq_tokens = seq_tokens_tmp

    if task_name in ['stability', 'fluorescence', 'pdbbind', 'skempi']: # 回归任务
        labels = pd.cut(labels, bins=num_bins, labels=False, ordered=True) # 离散化
    # change to indexs
    unique_labels = labels.unique()
    labels_to_indexs = {label: i for i, label in enumerate(unique_labels)}
    labels = labels.map(lambda x: labels_to_indexs[x])
    print("{} unique labels: {}".format(task_name.capitalize(), len(unique_labels)))
    # labels = labels.values.tolist()

    # Load preprocessed data
    if data_type != 'pt': # 'ft'
        pro_ids = load_tensor_for_pro_ids(os.path.join(args.dir_input, 'seqs_id_%s.txt' % data_type), torch.FloatTensor)
        goterms = load_tensor(os.path.join(args.dir_input, 'goterms_%s' % data_type), torch.FloatTensor)
        cath = CATHDataset(os.path.join(args.dir_input, 'structure_%s.jsonl' % data_type))
    else: # 'pt
        pro_ids = load_tensor_for_pro_ids(os.path.join(args.dir_input, 'seqs_id.txt'), torch.FloatTensor)
        goterms = load_tensor(os.path.join(args.dir_input, 'goterms'), torch.FloatTensor)
        cath = CATHDataset(os.path.join(args.dir_input, 'structure.jsonl'))

    unequal_count = 0
    for i in reversed(range(len(seq_tokens))):
        if len(seq_tokens[i]) != len(cath.data[i]['seq']) and (len(seq_tokens[i]) < args.max_pro_seq_len or len(cath.data[i]['seq']) < args.max_pro_seq_len):
            unequal_count += 1
            del seq_tokens[i]
            del cath.data[i]

    dataset = list(zip(seq_tokens, go_tokens, pro_ids, goterms, cath.data, labels))
    if task_name == 'kinase' or task_name == 'skempi':
        dataset = np.array(dataset)
        dataset = dataset[~pd.Series(go_tokens).isna()].tolist() # drop nan
        if task_name == 'kinase':
            dataset = stable_wrapper(shuffle_dataset(dataset, SEED), SEED)
            dataset = dataset[:5000]
    if shuffle:
        dataset = stable_wrapper(shuffle_dataset(dataset, SEED), SEED)
    
    if args.attn_vis and (slicer is not None) or (slicer[1]-slicer[0])>100: # 若是可视化注意力值，可精确选取某些指定数据；或OT调模型时截取
        dataset = dataset[slicer[0]: slicer[1]]

    dataset_pack, dataset_structure = transform_data(dataset, args.max_pro_seq_len, args.dir_input)
    dataset_tuple = (dataset_pack, dataset_structure)

    seqs = [d[0] for d in dataset] # 原始蛋白序列，作为attn map纵坐标
    pros = [d[2] for d in dataset] # 蛋白质名称索引

    te = time.time()
    print(f"Loading and transforming data cost time: {te-ts:.1f} secs")
    return dataset_tuple, seqs, pros


class Simplified_Trainer:
    """ Simplified trainer for OT basd model adaptation, utilize OT based measure as objective to optimize the model.
    """
    def __init__(self, model, args):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': args.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=args.lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch_size = args.batch_size
        self.device = args.device
        self.args = args

    def train(self, dataset_tuple, dir_input, data_type, src_embs_and_labels, loss_instance=nn.MSELoss()):
        src_embs = torch.tensor(np.stack(src_embs_and_labels['pro_emb'].tolist(), axis=0), dtype=torch.float32)
        src_labels = torch.tensor(np.stack(src_embs_and_labels['label'].tolist(), axis=0), dtype=torch.long)

        dataset, dataset_structure = dataset_tuple
        datasampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size)
        dataloaderstruc = torch_geometric.loader.DataListLoader(dataset_structure, num_workers=4, batch_size=self.batch_size)
        Loss = loss_instance # nn.MSELoss()
        frm_type = ''
        self.model.train()
        self.optimizer.zero_grad()
        spent_time_accumulation = 0
        accum_loss = .0
        src_batch_size = len(src_embs) // len(dataloader)
        for step, batch in enumerate(zip(dataloader, dataloaderstruc)):
            # slice batch src embs and labels
            # batch_src_embs = src_embs[step*self.batch_size: (step+1)*self.batch_size].to(self.device)
            # batch_src_labels = src_labels[step*self.batch_size: (step+1)*self.batch_size].to(self.device)
            batch_src_embs = src_embs[step*src_batch_size: (step+1)*src_batch_size].to(self.device)
            batch_src_labels = src_labels[step*src_batch_size: (step+1)*src_batch_size].to(self.device)

            # generate batch tgt training data
            start_time_batch = time.time()
            batch1, batch2 = batch
            gpu_split = torch.tensor(list(range(batch1[0].shape[0]))).to(self.device)
            seq_tokens, go_tokens, pro_ids, goterms, proteins_num, goterms_num, labels = batch1
            max_protein_len_batch = torch.max(proteins_num)
            max_goterms_len_batch = torch.max(goterms_num)
            seq_tokens, go_tokens = seq_tokens[:, :max_protein_len_batch], go_tokens[:, :max_goterms_len_batch]
            seq_tokens, go_tokens, labels = seq_tokens.to(self.device), go_tokens.to(self.device), labels.to(self.device)
            all_seqs_emb = []
            for pro_id in pro_ids:
                seq_emb_path = os.path.join(os.path.join(dir_input, 'sequences_emb_%s' % data_type), pro_id+".npy")
                one_seq_emb = np.load(seq_emb_path, allow_pickle=True)
                proteins_new = np.zeros((max_protein_len_batch, one_seq_emb.shape[1]))
                proteins_new[:one_seq_emb.shape[0], :] = one_seq_emb   
                all_seqs_emb.append(proteins_new.tolist())
            pro_seqs = torch.tensor(all_seqs_emb, dtype=torch.float32)
            pro_seqs = pro_seqs.to(self.device)
            pro_seqs, goterms = pro_seqs[:, :max_protein_len_batch, :], goterms[:, :max_goterms_len_batch, :]
            pro_seqs, goterms, proteins_num, goterms_num = \
                pro_seqs.to(self.device), goterms.to(self.device), proteins_num.to(self.device), goterms_num.to(self.device)
            batch2 = [batch_each_sample.to(self.device) for batch_each_sample in batch2]
            data_pack = (pro_seqs, batch2, gpu_split, goterms, proteins_num, goterms_num)

            # pass
            _, batch_tgt_embs = self.model(data_pack)
            # calculate loss
            # loss = Loss.closs(batch_tgt_embs, labels, batch_src_embs, batch_src_labels) # shift tgt to src
            # loss = Loss(batch_src_embs, batch_tgt_embs) # Loss(batch_tgt_embs, batch_src_embs)
            loss = Loss.cotfrm(batch_src_embs, batch_tgt_embs)
            accum_loss += loss.clone().detach().cpu().item()
            loss = loss / self.args.gradient_accumulation
            loss.backward()
            if (step+1) % self.args.gradient_accumulation == 0 or (step+1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            end_time_batch = time.time()
            seconds = end_time_batch-start_time_batch
            spent_time_accumulation += seconds

            f = open(self.args.logs_path, 'a', encoding='utf-8')
            if step % self.args.verbose_steps == 0 or (step+1)==len(dataloader):
                if frm_type != Loss.frm_type:
                    info_line = "Using {} OTFRM".format(Loss.frm_type.upper())
                    f.write(info_line + '\n')
                    print(info_line)
                    frm_type = Loss.frm_type
                
                avg_loss = accum_loss / (step+1)
                results_line = "Finish batch: %d/%d -- Avg-OTLoss: %.5f, accumulate time: %s" %  ( step, len(dataloader), avg_loss, 
                                str(datetime.timedelta(seconds=spent_time_accumulation)))
                f.write(results_line + '\n')
                print(results_line) # , end='\r'
                if avg_loss < 0.0:
                    print("Warning: Avg-Loss is negative, stop model adaption!")
                    f.write("Encounting negative avg-loss, early stopping model adaptation!")
                    break
        
        f.close()
        # save OT objective optimized model
        torch.save(self.model.state_dict(), self.args.ckpt_path+'_otloss{:.2f}_bs{}.pth'.format(accum_loss/(step+1), self.args.batch_size))
        print("Finish one epoch OT model optimization")
            


class Tester:
    def __init__(self, model, batch_size, device, get_attn=False):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.get_attn = get_attn

    def test(self, dataset_tuple, dir_input, data_type, wo_msa):
        self.model.eval()
        dataset, dataset_structure = dataset_tuple
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        dataloaderstruc = torch_geometric.loader.DataListLoader(dataset_structure, num_workers=4, batch_size=self.batch_size)

        all_attn_in_batches = []
        pro_emb_in_batches = []
        tqdm_bars = tqdm(zip(dataloader, dataloaderstruc), desc='Test single-forward', total=len(dataloader))
        for step, batch in stable_wrapper(enumerate(tqdm_bars), SEED):
            batch1, batch2 = batch
            gpu_split = torch.tensor(list(range(batch1[0].shape[0]))).to(self.device)
            seq_tokens, go_tokens, pro_ids, goterms, proteins_num, goterms_num, labels = batch1
            max_protein_len_batch = torch.max(proteins_num)
            max_goterms_len_batch = torch.max(goterms_num)
            seq_tokens, go_tokens = seq_tokens[:, :max_protein_len_batch], go_tokens[:, :max_goterms_len_batch]
            seq_tokens, go_tokens, labels = seq_tokens.to(self.device), go_tokens.to(self.device), labels.to(self.device)
                
            if not wo_msa:
                # 根据seq检索ESM-MSA-1b的seq embedding
                all_seqs_emb = []
                for pro_id in pro_ids:
                    if data_type != 'pt':
                        seq_emb_path = os.path.join(os.path.join(dir_input, 'sequences_emb_%s' % data_type), pro_id+".npy")
                    else:
                        seq_emb_path = os.path.join(os.path.join(dir_input, 'sequences_emb'), pro_id+".npy")
                    one_seq_emb = np.load(seq_emb_path, allow_pickle=True)
                    proteins_new = np.zeros((max_protein_len_batch, one_seq_emb.shape[1]))
                    proteins_new[:one_seq_emb.shape[0], :] = one_seq_emb   
                    all_seqs_emb.append(proteins_new.tolist())
                pro_seqs = torch.tensor(all_seqs_emb, dtype=torch.float32)
                pro_seqs = pro_seqs.to(self.device)
                pro_seqs = pro_seqs[:, :max_protein_len_batch, :]
            else:
                pro_seqs = seq_tokens

            goterms = goterms[:, :max_goterms_len_batch, :]
            pro_seqs, goterms, proteins_num, goterms_num = \
                pro_seqs.to(self.device), goterms.to(self.device), proteins_num.to(self.device), goterms_num.to(self.device)
            
            batch2 = [batch_each_sample.to(self.device) for batch_each_sample in batch2]
            data_pack = (pro_seqs, batch2, gpu_split, goterms, proteins_num, goterms_num)

            with torch.no_grad():
                if self.get_attn:
                    all_attn, pro_emb = self.model(data_pack)
                    all_attn_in_batches.append(all_attn)
                else:
                    _, pro_emb = self.model(data_pack)
                pro_emb_in_batches.append(pro_emb.detach().cpu().numpy())

        return all_attn_in_batches, pro_emb_in_batches


def get_args():
    """ 参数配置器 """
    parser = ArgumentParser(description='Model configuration')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=2, help='Id of gpu, set -1 to use cpu instead')
    parser.add_argument("--ft", type=str, default='multi', help='The type of finetuning, orig or multi', choices=['orig', 'multi'])
    args = parser.parse_args()
    
    args.device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else torch.device('cpu')
    
    config = yaml.load(open("./config.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    # config dict -> args
    for k, v in config.items():
        setattr(args, k, v)
    
    return args


def main(args):
    for task_name, task_config in args.tasks_config_hub.items():
        args.dir_input = task_config[0]
        TaskCombs = ['MLM+RMD', 'GO+RMD', 'MLM+GO+D', 'RMD']
        if task_name != 'kinase':
            # 除kinase外的数据集都有train, valid
            seq_txts = [task_config[1], task_config[2]]
            model_ckpts = task_config[3:]
        else:
            # kinase数据集没有valid
            seq_txts = [task_config[1]]
            model_ckpts = task_config[2:]
        for seq_txt in seq_txts:
            data_type =  seq_txt.split('.')[0].split('_')[-1] # test / valid / all' # pt'
            for i, ckpt in enumerate(model_ckpts):
                comb = TaskCombs[i]
                print("Processing: {}-{}-{} ".format(task_name.upper(), data_type.upper(), comb))
                # 生成测试集多模态输入数据
                testset_tuple, seqs, pros = generate_dataset(args, seq_txt, data_type, task_name, args.slicer)
                # named_model_dict = model.state_dict()
                # pretrain_model_dict = torch.load(args.pretrain_weights_pth, map_location=args.device)
                
                for scheme_type in ['finetune']: # ['wo_pretrain', 'pretrain', 'finetune']:
                    if scheme_type == 'wo_pretrain':
                        args.wo_msa, args.no_go = True, True
                    else:
                        args.wo_msa, args.no_go = False, False
                    
                    # 模型定义
                    model = Predictor(args)
                    if scheme_type in ['pretrain', 'finetune']:
                        # 预训练/微调模型加载
                        print("Loading {} model weights...".format(scheme_type))
                        # loaded_model_dict = torch.load(ckpt, map_location=args.device) if scheme_type=='finetune' else pretrain_model_dict
                        loaded_model_dict = torch.load(ckpt, map_location=args.device)
                        # loaded_model_dict = {k.replace('funsion', 'fusion'): v for k, v in loaded_model_dict.items() if 'funsion' in k}
                        if ('funsion' in model.state_dict().keys() and 'fusion' in loaded_model_dict.keys()) or \
                            ('fusion' in model.state_dict().keys() and 'funsion' in loaded_model_dict.keys()):
                            print("Fusion doesn't mactch !")
                        
                        matched_dict = {k: v for k, v in loaded_model_dict.items() if k in model.state_dict()} # 去除预训练模型多余的任务头
                        if scheme_type == 'wo_pretrain': # 无MSA-1b预训练embedding，需要初始化
                            matched_dict['embedding.weight'] = model.state_dict()['embedding.weight']
                        # matched_dict.update(loaded_model_dict)
                        model.load_state_dict(matched_dict, strict=True)
                    model.to(args.device)

                    # Forward pass
                    t0 = time.time()
                    tester = Tester(model, batch_size=args.batch_size if task_name!='kinase' else args.batch_size//2, device=args.device, get_attn=args.get_attn)
                    _, pro_emb_in_batches = tester.test(testset_tuple, args.dir_input, data_type=data_type, wo_msa=args.wo_msa)
                    
                    # Store embeddings
                    pro_embs_df = pd.DataFrame(columns=['pro_id', 'pro_seq', 'pro_emb'])
                    pro_embs_df['pro_id'] = pros
                    pro_embs_df['pro_seq'] = seqs
                    pro_embs_df['pro_emb'] = np.concatenate(pro_emb_in_batches, axis=0).tolist()
                    if args.ft == 'orig': # 一般emb生成
                        pro_embs_pth = os.path.join(args.pro_embs_base_pth, task_name)
                        os.makedirs(pro_embs_pth, exist_ok=True)
                        pro_embs_df.to_pickle(os.path.join(pro_embs_pth, f'{task_name}_{scheme_type}_pro_embs_{data_type}.pkl'))
                    elif args.ft == 'multi': # Multi-tasks combination
                        pro_embs_pth = os.path.join(args.pro_embs_base_pth, task_name)
                        os.makedirs(pro_embs_pth, exist_ok=True)
                        pro_embs_df.to_pickle(os.path.join(pro_embs_pth, '{}->{}_pretrain_pro_embs_{}.pkl'.format
                                            (comb, task_name, data_type)))

                    t1 = time.time()
                    print(f"cost time: {t1-t0:.2f} secs")
        print("=== " * 10)

if __name__ == '__main__':
    args = get_args()
    main(args)
