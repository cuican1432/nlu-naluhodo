from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch
import re
from tqdm import tqdm_notebook as tqdm
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from fastai.text import Tokenizer, Vocab
import pandas as pd
import collections
import os
import pdb
from tqdm import tqdm, trange
import sys
import random
import numpy as np
# import apex
from sklearn.model_selection import train_test_split
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam
import logging

# import BertForMultiLabelSequenceClassification
from dataRepresentation import InputExample, InputFeatures, DataProcessor, MultiLabelTextProcessor
from util import *
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from optimizer import CyclicLR

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

def fit(num_epocs = 2):
    global_step = 0
    model.train()
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
    #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = args['learning_rate'] * warmup_linear(global_step/t_total, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        eval()


def eval():
    args['output_dir'].mkdir(exist_ok=True)

    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])
    
    all_logits = None
    all_labels = None
    
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

#         logits = logits.detach().cpu().numpy()
#         label_ids = label_ids.to('cpu').numpy()
#         tmp_eval_accuracy = accuracy(logits, label_ids)
        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
            
        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:    
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
        

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
#     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
#               'loss': tr_loss/nb_tr_steps,
              'roc_auc': roc_auc  }

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
#             writer.write("%s = %s\n" % (key, str(result[key])))
    return result


if __name__ == '__main__':

	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
	                    datefmt='%m/%d/%Y %H:%M:%S',
	                    level=logging.INFO)
	logger = logging.getLogger(__name__)

	DATA_PATH = Path('/scratch/cc5048/bert/')/'beeap_1'
	DATA_PATH.mkdir(exist_ok = True)

	PATH = Path('/scratch/cc5048/bert')/'beeap_1'/'tmp'
	PATH.mkdir(exist_ok = True)

	CLAS_DATA_PATH = PATH/'class'
	CLAS_DATA_PATH.mkdir(exist_ok = True)

	model_state_dict = None

	# BERT_PRETRAINED_PATH = Path('../trained_model/')
	BERT_PRETRAINED_PATH = Path('/scratch/cc5048/bert')/'uncased_L-12_H-768_A-12'
	# BERT_PRETRAINED_PATH = Path('../../complaints/bert/pretrained-weights/cased_L-12_H-768_A-12/')
	# BERT_PRETRAINED_PATH = Path('../../complaints/bert/pretrained-weights/uncased_L-24_H-1024_A-16/')
	# BERT_FINETUNED_WEIGHTS = Path('../trained_model/toxic_comments')

	PYTORCH_PRETRAINED_BERT_CACHE = BERT_PRETRAINED_PATH/'cache/'
	PYTORCH_PRETRAINED_BERT_CACHE.mkdir(exist_ok = True)

	# output_model_file = os.path.join(BERT_FINETUNED_WEIGHTS, "pytorch_model.bin")

	# Load a trained model that you have fine-tuned

	# model_state_dict = torch.load(output_model_file)


	args = {
	    "train_size": -1,
	    "val_size": -1,
	    "full_data_dir": DATA_PATH,
	    "data_dir": PATH,
	    "task_name": "toxic_multilabel",
	    "no_cuda": False,
	    "bert_model": BERT_PRETRAINED_PATH,
	    "output_dir": CLAS_DATA_PATH/'output',
	    "max_seq_length": 128,
	    "do_train": True,
	    "do_eval": True,
	    "do_lower_case": True,
	    "train_batch_size": 32,
	    "eval_batch_size": 32,
	    "learning_rate": 3e-5,
	    "num_train_epochs": 4.0,
	    "warmup_proportion": 0.1,
	    "no_cuda": False,
	    "local_rank": -1,
	    "seed": 42,
	    "gradient_accumulation_steps": 1,
	    "optimize_on_cpu": False,
	    "fp16": False,
	    "loss_scale": 128
	}

	def warmup_linear(x, warmup=0.002):
	    if x < warmup:
	        return x/warmup
	    return 1.0 - x

	processors = {"toxic_multilabel": MultiLabelTextProcessor}
	task_name = args['task_name'].lower()
	random.seed(args['seed'])
	np.random.seed(args['seed'])
	torch.manual_seed(args['seed'])

	# Setup GPU parameters

	if args["local_rank"] == -1 or args["no_cuda"]:
	    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
	    n_gpu = torch.cuda.device_count()
	#     n_gpu = 1
	else:
	    torch.cuda.set_device(args['local_rank'])
	    device = torch.device("cuda", args['local_rank'])
	    n_gpu = 1
	    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
	    torch.distributed.init_process_group(backend='nccl')

	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
	        device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))

	args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

	if n_gpu > 0:
	    torch.cuda.manual_seed_all(args['seed'])

	if task_name not in processors:
	    raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name](args['data_dir'])
	label_list = processor.get_labels()
	num_labels = len(label_list)

	tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case = args['do_lower_case'])

	train_examples = None
	num_train_steps = None

	if args['do_train']:
	    train_examples = processor.get_train_examples(args['full_data_dir'], size=args['train_size'])
	#     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
	    num_train_steps = int(
	        len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])

	convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch('/scratch/cc5048/bert/uncased_L-12_H-768_A-12/bert_model.ckpt', '/scratch/cc5048/bert/uncased_L-12_H-768_A-12/bert_config.json',  \
		'/scratch/cc5048/bert/uncased_L-12_H-768_A-12/pytorch_model.bin')

	# Prepare model
	def get_model():
	#     pdb.set_trace()
	    if model_state_dict:
	        model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels = num_labels, state_dict=model_state_dict)
	    else:
	        model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels = num_labels)
	    return model

	model = get_model()

	if args['fp16']:
	    model.half()
	model.to(device)
	if args['local_rank'] != -1:
	    try:
	        from apex.parallel import DistributedDataParallel as DDP
	    except ImportError:
	        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

	    model = DDP(model)
	elif n_gpu > 1:
	    model = torch.nn.DataParallel(model)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
	    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	    ]
	t_total = num_train_steps
	if args['local_rank'] != -1:
	    t_total = t_total // torch.distributed.get_world_size()
	if args['fp16']:
	    try:
	        from apex.optimizers import FP16_Optimizer
	        from apex.optimizers import FusedAdam
	    except ImportError:
	        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

	    optimizer = FusedAdam(optimizer_grouped_parameters,
	                          lr=args['learning_rate'],
	                          bias_correction=False,
	                          max_grad_norm=1.0)
	    if args['loss_scale'] == 0:
	        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
	    else:
	        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args['loss_scale'])

	else:
	    optimizer = BertAdam(optimizer_grouped_parameters,
	                         lr=args['learning_rate'],
	                         warmup=args['warmup_proportion'],
	                         t_total=t_total)

	scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)
	eval_examples = processor.get_dev_examples(args['data_dir'], size=args['val_size'])
	train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_examples))
	logger.info("  Batch size = %d", args['train_batch_size'])
	logger.info("  Num steps = %d", num_train_steps)
	all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
	train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	if args['local_rank'] == -1:
	    train_sampler = RandomSampler(train_data)
	else:
	    train_sampler = DistributedSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])


	model.unfreeze_bert_encoder()
	fit(num_epocs=args['num_train_epochs'])
