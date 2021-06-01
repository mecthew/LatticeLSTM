# -*- coding:utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-07-06 11:08:27

import argparse
import sys
if sys.version > '3':   # python3
    import pickle
else:                   # python2
    import pickle as pickle
import copy
import gc
import os
import random
import sys
import time
import shutil
import datetime

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from model.bilstm_ple_crf import BiLSTM_CRF as PleSeqModel
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import ATTR_NULLKEY
from utils.data import Data
from utils.extract_entity import extract_kvpairs_in_bmoes
from utils.extract_entity import extract_kvpairs_by_start_end
from utils.metric import get_ner_fmeasure
from utils.logger import get_logger

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.detach().cpu().numpy()
    gold = gold_variable.detach().cpu().numpy()
    mask = mask_variable.detach().cpu().numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.detach().cpu().numpy()
    pred_tag = pred_variable.detach().cpu().numpy()
    gold_tag = gold_variable.detach().cpu().numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def convert_attr_seq_to_ner_seq(attr_start_pred, attr_end_pred, label_alphabet, attr_label_alphabet, tagscheme, gpu=True):
    batch_size = attr_start_pred.size(0)
    seq_len = attr_start_pred.size(1)
    attr_start_pred_tag = attr_start_pred.detach().cpu().numpy()
    attr_end_pred_tag = attr_end_pred.detach().cpu().numpy()
    pred_label = []
    pred_label_text = []
    for idx in range(batch_size):
        attr_start_seq, attr_end_seq = attr_start_pred_tag[idx], attr_end_pred_tag[idx]
        _fake_words = ['null'] * attr_start_seq.shape[0]
        pairs = extract_kvpairs_by_start_end(attr_start_seq, attr_end_seq, _fake_words, attr_label_alphabet.get_index(ATTR_NULLKEY))
        ner_seqs = ['O'] * seq_len
        # print("attr start: {}, {}".format(attr_start_seq.shape[0], [attr_label_alphabet.get_instance(at) for at in attr_start_seq]))
        # print("pairs: {}".format(pairs))
        # print("="*50, 'END')
        for pair in pairs:
            (spos, epos), attr, _ = pair
            attr_name = attr_label_alphabet.get_instance(attr)
            if attr_name is None:
                continue
            if tagscheme == 'BMES':
                if epos - spos == 1:
                    ner_seqs[spos] = 'S-' + attr_name
                else:
                    ner_seqs[spos] = 'B-' + attr_name
                    ner_seqs[epos-1] = 'E-' + attr_name
                    ner_seqs[spos+1: epos-1] = ['M-' + attr_name] * (epos - spos - 2)
            elif tagscheme == 'BIO':
                if epos - spos == 1:
                    ner_seqs[spos] = 'B-' + attr_name
                else:
                    ner_seqs[spos] = 'B-' + attr_name
                    ner_seqs[spos+1: epos-1] = ['I-' + attr_name] * (epos - spos - 2)
            else:
                raise ValueError('Unknown tagscheme: {}!'.format(tagscheme))
        try:
            unknown_idx = label_alphabet.get_index('O')     # 因为_tag是根据attr生成，可能不在label_alphabet里
            pred_label.append([label_alphabet.get_index(_tag) if _tag in label_alphabet.instances else unknown_idx for _tag in ner_seqs])
        except:
            # print("Error in {}".format(ner_seqs))
            print('Error')
        pred_label_text.append(ner_seqs)
    pred_variable = torch.LongTensor(pred_label)
    if gpu:
        pred_variable = pred_variable.cuda()
    return pred_variable


def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, new_tag_scheme):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    with torch.no_grad():
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen,\
                batch_charrecover, batch_label, mask, batch_span_label, batch_attr_start_label, batch_attr_end_label \
                = batchify_with_label(instance, data.HP_gpu, data.span_label_alphabet.get_index('O'),
                                      data.attr_label_alphabet.get_index(ATTR_NULLKEY), True)
            if new_tag_scheme:
                _, span_tag_seq, attr_start_output, attr_end_output \
                    = model.neg_log_likelihood_loss(gaz_list,
                                                    batch_word,
                                                    batch_biword,
                                                    batch_wordlen,
                                                    batch_char,
                                                    batch_charlen,
                                                    batch_charrecover,
                                                    batch_span_label,
                                                    batch_attr_start_label,
                                                    batch_attr_end_label,
                                                    mask)
                tag_seq = convert_attr_seq_to_ner_seq(attr_start_output, attr_end_output, data.label_alphabet,
                                                      data.attr_label_alphabet, data.tagScheme)
            else:
                tag_seq = model(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                                mask)
                # print "tag:",tag_seq
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            pred_results += pred_label
            gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results


def batchify_with_label(input_batch_list, gpu, span_label_pad, attr_label_pad, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    span_labels = [sent[5] for sent in input_batch_list]
    attr_start_labels = [sent[6] for sent in input_batch_list]
    attr_end_labels = [sent[7] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().numpy()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    biword_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    span_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    attr_start_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    attr_end_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        span_label_seq_tensor[idx, :seqlen] = torch.LongTensor(span_labels)
        attr_start_label_seq_tensor[idx, :seqlen] = torch.LongTensor(attr_start_labels)
        attr_end_label_seq_tensor[idx, :seqlen] = torch.LongTensor(attr_end_labels)
        mask[idx, :seqlen] = torch.ones((seqlen,), dtype=torch.bool)  #torch.ByteTensor([1] * seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    ## not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    span_label_seq_tensor = span_label_seq_tensor[word_perm_idx]
    attr_start_label_seq_tensor = attr_start_label_seq_tensor[word_perm_idx]
    attr_end_label_seq_tensor = attr_end_label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long)
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    ## keep the gaz_list in orignial order

    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        span_label_seq_tensor = span_label_seq_tensor.cuda()
        attr_start_label_seq_tensor = attr_start_label_seq_tensor.cuda()
        attr_end_label_seq_tensor = attr_end_label_seq_tensor.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, \
           span_label_seq_tensor, attr_start_label_seq_tensor, attr_end_label_seq_tensor


def train(data, save_model_dir, save_dset_path, use_ple_lstm, seg=True, epochs=100, new_tag_scheme=False):
    print("Training model...")
    data.show_data_summary()
    save_data_setting(data, save_dset_path)
    if new_tag_scheme:  # 使用多任务标注方案
        model = PleSeqModel(data, use_ple_lstm=use_ple_lstm)
        model.to(device)
    else:
        model = SeqModel(data)
        model.to(device)
    print("finished built model.")
    loss_function = nn.NLLLoss()
    parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    optimizer = optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    best_dev = -1
    data.HP_iteration = epochs
    best_model_name = None
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print(("Epoch: %s/%s" % (idx, data.HP_iteration)))
        # optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = 1  ## current only support batch size = 1 to compulate and accumulate to data.HP_batch_size update weights
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            word_text = [[data.word_alphabet.get_instance(l) for l in sample[0]] for sample in instance]
            label_text = [[data.label_alphabet.get_instance(l) for l in sample[4]] for sample in instance]
            # print("="*30, 'Gold')
            # print(word_text)
            # print(len(label_text[0]), label_text)
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, \
                batch_span_label, batch_attr_start_label, batch_attr_end_label \
                = batchify_with_label(instance, data.HP_gpu, data.span_label_alphabet.get_index('O'),
                                      data.attr_label_alphabet.get_index(ATTR_NULLKEY))
            # print "gaz_list:",gaz_list
            # exit(0)

            instance_count += 1
            if new_tag_scheme:
                loss, span_tag_seq, attr_start_output, attr_end_output \
                    = model.neg_log_likelihood_loss(gaz_list,
                                                    batch_word,
                                                    batch_biword,
                                                    batch_wordlen,
                                                    batch_char,
                                                    batch_charlen,
                                                    batch_charrecover,
                                                    batch_span_label,
                                                    batch_attr_start_label,
                                                    batch_attr_end_label,
                                                    mask)
                tag_seq = convert_attr_seq_to_ner_seq(attr_start_output, attr_end_output, data.label_alphabet,
                                                      data.attr_label_alphabet, data.tagScheme)
            else:
                loss, tag_seq = model.neg_log_likelihood_loss(gaz_list,
                                                              batch_word,
                                                              batch_biword,
                                                              batch_wordlen,
                                                              batch_char,
                                                              batch_charlen,
                                                              batch_charrecover,
                                                              batch_label,
                                                              mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
            end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token)))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss)))
        # exit(0)
        # continue
        speed, acc, p, r, f, _ = evaluate(data, model, "dev", new_tag_scheme)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
        else:
            current_score = acc

        is_best_model = False
        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = save_model_dir + '/latticelstm' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
            best_model_name = model_name
            is_best_model = True
            # ## decode test
        logger.info("Is best model: {}".format(is_best_model))
        print("Is best model: {}".format(is_best_model))
        if seg:
            print(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f)))
            logger.info(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f)))
        else:
            print(("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc)))
            logger.info(("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc)))

        if args.dataset != 'msra':
            speed, acc, p, r, f, _ = evaluate(data, model, "test", new_tag_scheme)
            test_finish = time.time()
            test_cost = test_finish - dev_finish

            if seg:
                print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                    test_cost, speed, acc, p, r, f)))
                logger.info(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                    test_cost, speed, acc, p, r, f)))
            else:
                print(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc)))
                logger.info(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc)))
        gc.collect()
    if best_model_name:
        shutil.copy(best_model_name, save_model_dir + '/best.model')


def load_model_decode(model_dir, data, name, gpu, seg=True, use_ple_lstm=False, new_tag_scheme=False):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    if new_tag_scheme:
        model = PleSeqModel(data, use_ple_lstm)
    else:
        model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    # model = torch.load(model_dir)

    print(("Decode %s data ..." % (name)))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name, new_tag_scheme)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f)))
        logger.info(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f)))
    else:
        print(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc)))
        logger.info(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc)))
    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    # parser.add_argument('--savemodel', default="output/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes")
    parser.add_argument('--dev', default="data/conll03/dev.bmes")
    parser.add_argument('--test', default="data/conll03/test.bmes")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    parser.add_argument('--dataset')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--new_tag_scheme', default=0, type=int)
    parser.add_argument('--latticelstm_num', default=1, type=int, help="主要用在判断是否使用多个latticelstm输出作为ple输入")
    parser.add_argument('--char_emb', type=str, default="data/gigaword_chn.all.a2b.uni.11k.50d.vec")
    parser.add_argument('--gaz_file', type=str, default="data/ctb.704k.50d.vec")
    parser.add_argument('--use_ple_lstm', action='store_true')

    args = parser.parse_args()
    log_dir = f'./output/logs/{args.dataset}/tagscheme{args.new_tag_scheme}'
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(sys.argv, log_dir + '/{}.log'.format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')))
    logger.info('Arguments:')
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    load_model_path = args.loadmodel if args.loadmodel else f'./output/ckpt/{args.dataset}/tagscheme{args.new_tag_scheme}/best.model'
    dset_dir = args.savedset if args.savedset else "data/{}/{}.dset".format(args.dataset, args.dataset)
    # os.makedirs(dset_dir, exist_ok=True)
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = f'./output/ckpt/{args.dataset}/tagscheme{args.new_tag_scheme}'
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    # if not os.path.exists('./data/ckpt'):
    #     os.makedirs('./data/ckpt')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    char_emb = args.char_emb
    bichar_emb = None
    gaz_file = args.gaz_file
    # gaz_file = None
    # char_emb = None
    # bichar_emb = None

    print("CuDNN:", torch.backends.cudnn.enabled)
    # gpu = False
    print("GPU available:", gpu)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:", gaz_file)
    print("Latticelstm num:", args.latticelstm_num)
    print("Use PLE lstm:", args.use_ple_lstm)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        data = Data(args.new_tag_scheme, args.dataset)
        data.HP_gpu = gpu
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.use_bigram = False
        data.gaz_dropout = 0.5
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False
        data.latticelstm_num = args.latticelstm_num
        data_initialization(data, gaz_file, train_file, dev_file, test_file)
        data.generate_instance_with_gaz(train_file, 'train')
        data.generate_instance_with_gaz(dev_file, 'dev')
        data.generate_instance_with_gaz(test_file, 'test')
        data.build_word_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        train(data, save_model_dir, dset_dir, use_ple_lstm=args.use_ple_lstm, seg=seg, epochs=args.epochs, new_tag_scheme=args.new_tag_scheme)
    elif status == 'test':
        data = load_data_setting(dset_dir)
        # data.generate_instance_with_gaz(dev_file, 'dev')
        # load_model_decode(model_dir, data, 'dev', gpu, seg)
        data.generate_instance_with_gaz(test_file, 'test')
        test_preds = load_model_decode(load_model_path, data, 'test', gpu, seg, args.use_ple_lstm, args.new_tag_scheme)
        test_texts = data.test_texts
        words = [instance[0] for instance in test_texts]
        labels = [instance[4] for instance in test_texts]
        label_alphabet = data.label_alphabet
        case_result = []
        for word_seq, gold_seq, pred_seq in zip(words, labels, test_preds):
            gold_pair = extract_kvpairs_in_bmoes(gold_seq, word_seq)
            pred_pair = extract_kvpairs_in_bmoes(pred_seq, word_seq)
            case_result.append((''.join(word_seq), str(gold_pair), str(pred_pair)))

        os.makedirs('./case_study', exist_ok=True)
        case_output_path = './case_study/latticelstm_{}_scheme{}.casestudy'.format(args.dataset, args.new_tag_scheme)
        case_fout = open(case_output_path, 'w')
        print("Saving case result to {}".format(case_output_path))
        logger.info("Saving case result to {}".format(case_output_path))
        for word_seq, gold_pair, pred_pair in case_result:
            case_fout.write(word_seq + '\n' + gold_pair + '\n' + pred_pair + '\n\n')

    elif status == 'decode':
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(raw_file, 'raw')
        decode_results = load_model_decode(load_model_path, data, 'raw', gpu, seg, args.use_ple_lstm, args.new_tag_scheme)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
