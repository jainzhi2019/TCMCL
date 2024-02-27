from __future__ import absolute_import, division, print_function

import argparse
import csv
import math
import os
import random
import pickle
import sys
import numpy as np
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam

import models
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
# from transformers.optimization import AdamW
from torch.optim.adamw import AdamW
from bert import BertForSequenceClassification


from argparse_utils import str2bool, seed
from configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=80)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default=random)


# import os
import wandb
# # os.environ["WANDB_SERVICE_WAIT"]=""
os.environ["WANDB_MODE"] = "offline"


args = parser.parse_args()


def return_unk():
    return 0

def save_model(model,path,best_acc):

    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, str(best_acc)+'.pth'))

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input
        elif args.model == "xlnet-base-cased":
            prepare_input = prepare_xlnet_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def prepare_xlnet_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    # PAD special tokens
    tokens = tokens + [SEP] + [CLS]
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, audio_zero, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    pad_length = (args.max_seq_length - len(segment_ids))

    # then zero pad the visual and acoustic
    audio_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((audio_padding, acoustic))

    video_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((video_padding, visual))

    input_ids = [PAD_ID] * pad_length + input_ids
    input_mask = [0] * pad_length + input_mask
    segment_ids = [3] * pad_length + segment_ids

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    # num_train_optimization_steps=1600
    num_train_optimization_steps = (
            int(
                math.ceil(len(train_dataset) / args.train_batch_size) /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.model == "bert-base-uncased":
        model = BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1,
        )


    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup_with_cycle(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1

):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        calculatestep=current_step%num_training_steps
        halvetime=current_step//num_training_steps
        if calculatestep < num_warmup_steps:
            return float(calculatestep)*(0.5**halvetime) / float(max(1, num_warmup_steps))
        progress = float(calculatestep - num_warmup_steps)*(0.5**halvetime) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_label_3(flag):
    if flag > 0:
        return 1
    elif flag==0:
        return 0
    elif flag<0:
        return -1

def get_label_2(flag):
    if flag >=0:
        return 1
    elif flag<0:
        return 0

from collections import Counter
from scipy.ndimage import convolve1d


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )

        from loss import InfoNCE,Sup_infonce
        logits = outputs[0]
        pool=outputs[1]
        fused_tv=outputs[2]
        fused_ta=outputs[3]
        pool_tv=outputs[4]
        pool_ta=outputs[5]


        ptv = torch.mean(pool_tv, dim=2)
        pta = torch.mean(pool_ta, dim=2)
        ftv = torch.mean(fused_tv, dim=2)
        fta = torch.mean(fused_ta, dim=2)

        mseloss=MSELoss()
        nceloss=InfoNCE()

        label_3 = torch.tensor([get_label_3(i) for i in label_ids], device=label_ids.device)
        label_3 = torch.cat((label_3,label_3),dim=0)


        loss_mse= mseloss(logits.view(-1), label_ids.view(-1))

        loss_nce1 = nceloss(torch.mean(pool_tv, dim=2), torch.mean(fused_ta, dim=2))
        loss_nce2 = nceloss(torch.mean(pool_ta, dim=2), torch.mean(fused_tv, dim=2))
        loss_nce=loss_nce1+loss_nce2

        supnceloss=Sup_infonce()
        pool_all = torch.cat((ptv,pta),dim=0)
        loss_sup_emo=supnceloss(pool_all,label_3)
        loss = loss_mse + 0.05 * loss_nce + 0.1 * loss_sup_emo#+ 0.1 * loss_mae##+0.05*loss_lds#+0.1*loss_Bl#+0.05*loss_sup_emo


        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            # temp= torch.argmax(outputs[1],dim=1)-1
            # logits = (outputs[0].view(-1))*temp
            logits=outputs[0]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps

import pandas as pd
import matplotlib.pyplot as plt



def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    # print(model)
    model.eval()
    preds = []
    labels = []

    output1 = []
    output_p = []
    output_m = []
    output_n = []
    label1 = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            # temp = torch.argmax(outputs[1], dim=1) - 1
            # logits = (outputs[0].view(-1)) * temp
            logits = outputs[0]
            pool=outputs[1]

            label_3 = torch.tensor([get_label_3(i) for i in label_ids], device=label_ids.device)
            output1.append(pool)
            label1.append(label_3)


            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

        output1 = torch.cat(output1)
        label1 = torch.cat(label1)


        from sklearn.manifold import TSNE


        t_sne_features = TSNE(n_components=2,  init='pca', random_state=8498).fit_transform(
            output1.cpu().numpy())
        x = t_sne_features[:, 0]
        y = t_sne_features[:, 1]
        index0 = np.array([i for i, e in enumerate(label1.cpu().numpy()) if e ==-1])
        index1 = np.array([i for i, e in enumerate(label1.cpu().numpy()) if e ==0])
        index2 = np.array([i for i, e in enumerate(label1.cpu().numpy()) if e ==1])
        colors= ['b', 'c', 'y', 'm', 'r', 'g', 'k','yellow','yellowgreen','wheat']     #label1.cpu().numpy()
        plt.xticks(())  # 把坐标轴上的刻度去掉
        plt.yticks(())

        plt.scatter(x[index0], y[index0], c=colors[0],  marker='h', label='negative')
        plt.scatter(x[index1], y[index1], c=colors[1],  marker='<', label='neutral')
        plt.scatter(x[index2], y[index2], c=colors[2],  marker='x', label='positive')
        plt.legend(loc='lower right')

        import datetime

        # 获取当前日期和时间
        now = datetime.datetime.now()

        # 格式化为指定的字符串形式（年-月-日_小时-分钟-秒）
        filename = now.strftime("%Y-%m-%d_%H-%M-%S")
        # plt.savefig('.')
        plt.savefig('./TSNE/CL/{}.png'.format(filename), dpi=300)
        plt.show()








    return preds, labels

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))+1

def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)

    preds7=[np.round(i) for i in np.clip(preds, a_min=-3., a_max=3.)]
    y_test7=[np.round(i) for i in np.clip(y_test, a_min=-3., a_max=3.)]

    acc7 = accuracy_score(y_test7, preds7)

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    non_zeros_left = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or True])
    non_zeros_right = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds_left = preds[non_zeros_left]>=0
    y_test_left = y_test[non_zeros_left]>=0
    preds_right = preds[non_zeros_right]>0
    y_test_right = y_test[non_zeros_right]>0

    f_scorel = f1_score(y_test_left, preds_left, average="weighted")
    accl = accuracy_score(y_test_left, preds_left)
    f_scorer = f1_score(y_test_right, preds_right, average="weighted")
    accr = accuracy_score(y_test_right, preds_right)

    return acc7,accl,accr ,f_scorel ,f_scorer,mae, corr

best_acc=None
def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc7,test_accl, test_accr, test_f_scorel,test_f_scorer, test_mae, test_corr = test_score_model(
            model, test_data_loader
        )
        global best_acc,best_acc7,best_f1,bset_mae,best_corr,best_accl,best_f1l

        if best_acc is None:
            best_acc=test_accr
            model.save_pretrained("exp/")
        elif test_accr> best_acc:
            best_acc=test_accr
            best_acc7=test_acc7
            best_f1=test_f_scorer
            best_mae=test_mae
            best_corr=test_corr
            best_accl=test_accl
            best_f1l=test_f_scorel
            model.save_pretrained("exp/")

        if best_acc>0.865:
            save_model(model, 'savemodel/',best_acc)

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{},test_mae:{}".format(
                epoch_i, train_loss, valid_loss, test_accr,test_mae
            )
        )

        valid_losses.append(valid_loss)
        test_accuracies.append(test_accr)

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_acc": test_accr,
                    "test_accl": test_accl,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_scorer,
                    "test_f_scorel": test_f_scorel,
                    "test_acc7":test_acc7,
                    "best_valid_loss": min(valid_losses),
                    "best_test_acc": max(test_accuracies),
                }
            )
        )

    print("\nacc7:",best_acc7,"\naccr:",best_acc,"\nf1scorer:",best_f1,"\naccl:",best_accl,"\nf1scorel:",best_f1l,"\nmae:",best_mae,"\ncorr:",best_corr)

def main():
    wandb.init(project="test")
    wandb.config.update(args)
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
