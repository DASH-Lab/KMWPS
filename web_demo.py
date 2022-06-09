#!/usr/bin/env python
# coding: utf-8

# from model.tiny_transformer import *
from utils import *
from dataloader import *
#import pickle5 as pickle
import pickle
import pandas as pd
from prerequisite import *
from preprocess import *
from torch.utils.data import Dataset
from prerequisite import *
from transformers import BertConfig
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertTokenizer, BertPreTrainedModel, BertModel#, 
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
import gradio as gr

import argparse

parser = argparse.ArgumentParser(description='Script of Brid call')
parser.add_argument('--gpu', '-gpu', type=str, default='0', help='gpu')
parser.add_argument('--hidden_layer', '-hl', type=int, default=12)
parser.add_argument('--hidden_size', '-hs', type=int, default=768)
parser.add_argument('--intermediate_size', '-is', type=int, default=3072)
parser.add_argument('--model_pth_name', '-mp', type=str, default="./saved_models/n4_h252_i_1024_head_12_kobert_exp-gradio/best_model.pth" )

parser.add_argument('--num_decoder_layer', '-ndl', type=int, default=1)
parser.add_argument('--decoder_head', '-dh', type=int, default=12)



args = parser.parse_args()

class config:
    # ---- factor ---- #
    debug = False
    full_cv = False
    mode = 'train'
    gpu = args.gpu
    dropout = 0.1
    heads = args.decoder_head  # 4
    encoder_layers = args.num_decoder_layer
    decoder_layers = args.num_decoder_layer
    d_model = 768
    d_ff = 1024  # 256
    batch_size = 1
    embedding = 'bert' # ['bert','roberta']
    emb_name = 'monologg/kobert'
    mawps_vocab = True

    max_length = 50  # 30
    vocab_size = 30000  # 30000
    init_range = 0.08  # 'Initialization range for seq2seq model'
    freeze_emb = True
    
    # -- encoder -- #
    num_hidden_layers = args.hidden_layer 
    hidden_size       = args.hidden_size #768#252
    intermediate_size = args.intermediate_size #3072#786
    model_path = args.model_pth_name
    # ---- Else ---- #
    num_workers = 8
    seed = 92

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu



def build_model(config, voc1, voc2, device, teacher=False):
    """
        Args:
            config (dict): command line arguments
            voc1 (object of class Voc1): vocabulary of source
            voc2 (object of class Voc2): vocabulary of target
            device (torch.device): GPU device
        Returns:
            model (object of class TransformerModel): model
    """

    model = TransformerModel(config, voc1, voc2, device, teacher=teacher)
    model = model.to(device)
    # print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters())}")


    return model
class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.cls = BertPreTrainingHeads(config)#, self.bert.embeddings.word_embeddings.weight)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        #self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, labels=None):
        outputs =  self.bert(input_ids, token_type_ids, attention_mask, output_attentions=True)
        sequence_output, att_output, pooled_output = outputs['last_hidden_state'], outputs['attentions'], outputs['pooler_output']

        # 추가
        prediction_logits, _  = self.cls(sequence_output, pooled_output)
        #
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return prediction_logits, att_output, sequence_output

# model
class BertEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', device='cuda', cfg=None, teacher=False, freeze_bert=False):
        super(BertEncoder, self).__init__()
        # self.bert_layer = TinyBertForPreTraining(BertConfig(num_hidden_layers=12, 
        #                                                         hidden_size=768,intermediate_size=3072)).from_pretrained(bert_model)
        if teacher:
            self.bert_layer = TinyBertForPreTraining(BertConfig(num_hidden_layers=12, 
                                                                hidden_size=768, intermediate_size=3072)).from_pretrained(bert_model)

        else: # student
            self.bert_layer = TinyBertForPreTraining(BertConfig(num_hidden_layers=cfg.num_hidden_layers, 
                                                            hidden_size=cfg.hidden_size,intermediate_size=cfg.intermediate_size, 
                                                            num_attention_heads=cfg.heads))#.from_pretrained(bert_model)

        #BertConfig(num_hidden_layers=4, hidden_size=252, intermediate_size=786)
        #BertConfig(num_hidden_layers=12, hidden_size=768, intermediate_size=3072)

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        self.device = device
        self.cfg = cfg

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

    def bertify_input(self, sentences):
        '''
        Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids

        Args:
            sentences (list): source sentences
        Returns:
            token_ids (tensor): tokenized sentences | size: [BS x S]
            attn_masks (tensor): masks padded indices | size: [BS x S]
            input_lengths (list): lengths of sentences | size: [BS]
        '''

        # Tokenize the input sentences for feeding into BERT
        all_tokens = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

        # Pad all the sentences to a maximum length
        input_lengths = [len(tokens) for tokens in all_tokens]
        max_length = max(input_lengths)
        max_length = self.cfg.max_length
        padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

        # Convert tokens to token ids
        token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(
            self.device)

        # Obtain attention masks
        pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
        attn_masks = (token_ids != pad_token).long()

        return token_ids, attn_masks, input_lengths

    def forward(self, sentences):
        '''
        Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token

        Args:
            sentences (list): source sentences
        Returns:
            cont_reps (tensor): BERT Embeddings | size: [BS x S x d_model]
            token_ids (tensor): tokenized sentences | size: [BS x S]
        '''

        # Preprocess sentences
        token_ids, attn_masks, input_lengths = self.bertify_input(sentences)

        cont_reps, attn, seq = self.bert_layer(token_ids, attention_mask=attn_masks) # logit, attn, embedding


        return cont_reps, attn, seq, token_ids


class RobertaEncoder(nn.Module):
    def __init__(self, roberta_model='roberta-base', device='cuda:0 ', freeze_roberta=False):
        super(RobertaEncoder, self).__init__()
        # self.roberta_layer = RobertaModel.from_pretrained(roberta_model, return_dict=False)
        # self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)

        self.roberta_layer = AutoModel.from_pretrained(roberta_model)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.device = device

        if freeze_roberta:
            for p in self.roberta_layer.parameters():
                p.requires_grad = False

    def robertify_input(self, sentences):
        '''
        Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids

        Args:
            sentences (list): source sentences
        Returns:
            token_ids (tensor): tokenized sentences | size: [BS x S]
            attn_masks (tensor): masks padded indices | size: [BS x S]
            input_lengths (list): lengths of sentences | size: [BS]
        '''

        # Tokenize the input sentences for feeding into RoBERTa
        all_tokens = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]

        # Pad all the sentences to a maximum length
        input_lengths = [len(tokens) for tokens in all_tokens]
        max_length = max(input_lengths)
        padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

        # Convert tokens to token ids
        token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(
            self.device)

        # Obtain attention masks
        pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
        attn_masks = (token_ids != pad_token).long()

        return token_ids, attn_masks, input_lengths

    def forward(self, sentences):
        '''
        Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token

        Args:
            sentences (list): source sentences
        Returns:
            cont_reps (tensor): RoBERTa Embeddings | size: [BS x S x d_model]
            token_ids (tensor): tokenized sentences | size: [BS x S]
        '''

        # Preprocess sentences
        token_ids, attn_masks, input_lengths = self.robertify_input(sentences)

        # Feed through RoBERTa
        cont_reps, _ = self.roberta_layer(token_ids, attention_mask=attn_masks).values()

        return cont_reps, token_ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))  # nn.Parameter causes the tensor to appear in the model.parameters()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # max_len x 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                -math.log(10000.0) / d_model))  # torch.arange(0, d_model, 2) gives 2i
        pe[:, 0::2] = torch.sin(position * div_term)  # all alternate columns 0 onwards
        pe[:, 1::2] = torch.cos(position * div_term)  # all alternate columns 1 onwards
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
            Args:
                x (tensor): embeddings | size : [max_len x batch_size x d_model]
            Returns:
                z (tensor) : embeddings with positional encoding | size : [max_len x batch_size x d_model]
        '''

        # print(x.shape)
        # print(self.scale.shape)
        # print(self.pe[:x.size(0), :].shape)
        x = x[:, :, :768] + self.scale * self.pe[:x.size(0), :]
        z = self.dropout(x)
        return z


class TransformerModel(nn.Module):
    def __init__(self, config, voc1, voc2, device,teacher, EOS_tag='</s>', SOS_tag='<s>'):
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = device
        self.voc1 = voc1
        self.voc2 = voc2
        self.EOS_tag = EOS_tag
        self.SOS_tag = SOS_tag
        self.EOS_token = voc2.get_id(EOS_tag)
        self.SOS_token = voc2.get_id(SOS_tag)

        if self.config.embedding == 'bert':
            config.d_model = 768
            self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config, teacher,self.config.freeze_emb)
        elif self.config.embedding == 'roberta':
            config.d_model = 768
            self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
        else:
            self.embedding1 = nn.Embedding(self.voc1.nwords, self.config.d_model)
            nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

        self.pos_embedding1 = PositionalEncoding(self.config.d_model, self.config.dropout)
        self.embedding2 = nn.Embedding(self.voc2.nwords, self.config.d_model)
        nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)

        self.pos_embedding2 = PositionalEncoding(self.config.d_model, self.config.dropout)

        self.transformer = nn.Transformer(d_model=self.config.d_model, nhead=self.config.heads,
                                          num_encoder_layers=self.config.encoder_layers,
                                          num_decoder_layers=self.config.decoder_layers,
                                          dim_feedforward=self.config.d_ff, dropout=self.config.dropout)

        self.fc_out = nn.Linear(self.config.d_model, self.voc2.nwords)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        '''
            Args:
                sz (integer): max_len of sequence in target without EOS i.e. (T-1)
            Returns:
                mask (tensor) : square mask | size : [T-1 x T-1]
        '''

        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        '''
            Args:
                inp (tensor): input indices | size : [S x BS]
            Returns:
                mask (tensor) : pad mask | size : [BS x S]
        '''

        mask = (inp == -1).transpose(0, 1)
        return mask

    def forward(self, ques, src, trg):
        '''
            Args:
                ques (list): raw source input | size : [BS]
                src (tensor): source indices | size : [S x BS]
                trg (tensor): target indices | size : [T x BS]
            Returns:
                output (tensor) : Network output | size : [T-1 x BS x voc2.nwords]
        '''

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
            src, attn, seq_emb, src_tokens = self.embedding1(ques)
            #             print('train src shape:',src.shape)
            src = src.transpose(0, 1)
            # src: Tensor [S x BS x d_model]
            src_pad_mask = self.make_len_mask(src_tokens.transpose(0, 1))
            src = self.pos_embedding1(src)
        else:
            src_pad_mask = self.make_len_mask(src)
            src = self.embedding1(src)
            src = self.pos_embedding1(src)

        trg_pad_mask = self.make_len_mask(trg)
        trg = self.embedding2(trg)
        trg = self.pos_embedding2(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
                                  memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask)

        output = self.fc_out(output)

        return output, src, attn, seq_emb

    def greedy_decode(self, ques=None, input_seq1=None, input_seq2=None, input_len2=None, criterion=None,
                      validation=False):
        '''
            Args:
                ques (list): raw source input | size : [BS]
                input_seq1 (tensor): source indices | size : [S x BS]
                input_seq2 (tensor): target indices | size : [T x BS]
                input_len2 (list): lengths of targets | size: [BS]
                validation (bool): whether validate
            Returns:
                if validation:
                    validation loss (float): Validation loss
                    decoded_words (list): predicted equations | size : [BS x target_len]
                else:
                    decoded_words (list): predicted equations | size : [BS x target_len]
        '''

        with torch.no_grad():
            loss = 0.0

            if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
                src, _ , _, _= self.embedding1(ques)
                src = src.transpose(0, 1)
                # src: Tensor [S x BS x emb1_size]
                memory = self.transformer.encoder(self.pos_embedding1(src))
            #                 print('memory:',memory.shape)
            else:
                memory = self.embedding1(input_seq1)
                memory = self.transformer.encoder(self.pos_embedding1(memory))

            # memory: S x BS x d_model

            input_list = [[self.SOS_token for i in range(input_seq1.size(1))]]

            decoded_words = [[] for i in range(input_seq1.size(1))]

            if validation:
                target_len = max(input_len2)
            else:
                target_len = self.config.max_length

            for step in range(target_len):
                decoder_input = torch.LongTensor(input_list).to(self.device)  # seq_len x bs

                decoder_output = self.fc_out(
                    self.transformer.decoder(self.pos_embedding2(self.embedding2(decoder_input)),
                                             memory))  # seq_len x bs x voc2.nwords

                if validation:
                    loss += criterion(decoder_output[-1, :, :], input_seq2[step])

                out_tokens = decoder_output.argmax(2)[-1, :]  # bs

                for i in range(input_seq1.size(1)):
                    if out_tokens[i].item() == self.EOS_token:
                        continue
                    decoded_words[i].append(self.voc2.get_word(out_tokens[i].item()))

                input_list.append(out_tokens.detach().tolist())

            if validation:
                return loss / target_len, decoded_words
            else:
                return decoded_words

    def sim_forward(self, ques=None, input_seq1=None):

        with torch.no_grad():
            loss = 0.0

            if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
                src, _ = self.embedding1(ques)
                src = src.transpose(0, 1)
                # src: Tensor [S x BS x emb1_size]
                memory = self.transformer.encoder(self.pos_embedding1(src))
            else:
                memory = self.transformer.encoder(self.pos_embedding1(self.embedding1(input_seq1)))
            # memory: S x BS x d_model

            input_list = [[self.SOS_token for i in range(input_seq1.size(1))]]

            decoded_words = [[] for i in range(input_seq1.size(1))]

            target_len = self.config.max_length

            for step in range(target_len):
                decoder_input = torch.LongTensor(input_list).to(self.device)  # seq_len x bs

                decoder_output = self.fc_out(
                    self.transformer.decoder(self.pos_embedding2(self.embedding2(decoder_input)),
                                             memory))  # seq_len x bs x voc2.nwords

                out_tokens = decoder_output.argmax(2)[-1, :]  # bs

                for i in range(input_seq1.size(1)):
                    if out_tokens[i].item() == self.EOS_token:
                        continue
                    decoded_words[i].append(self.voc2.get_word(out_tokens[i].item()))

                input_list.append(out_tokens.detach().tolist())

            return decoded_words


# In[ ]:


###############데이터 상수바꾸기위해#####################################################
# 이름들 전부 name0~ 이런식으로 바꾸기


name_ = ['민영', '유나', '정국', '유정', '태형', '남준', '윤기', '호석', '지민', '석진', '은지',

        '동희', '새별', '진범', '민하', '광수', '재석', '지효', '쯔위', '쯔양', '태준', '진솔', '지영', '건우', '송찬',

         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',

         '\(가\)', '\(나\)', '\(다\)', '\(라\)', '\(마\)', '\(바\)', '\(사\)', '\(아\)', '\(자\)', '\(차\)', '\(카\)', '\(타\)',
         '\(파\)', '\(하\)'

                  '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일',

         '농구공', '배구공', '테니스공', '탁구공', '야구공', '축구공',

         '노란색', '파란색', '빨간색', '주황색', '남색', '보라색', '흰색', '검은색', '초록색',

         # 다리 수 세려고 일부러 다뺐음
         #  '강아지', '개구리', '거위', '고라니', '고래', '고양이', '곰', '기린',  '늑대', '달팽이', '물고기',
         #  '병아리',  '비둘기',  '사자',  '여우', '오리',  '원숭이', '코끼리',  '토끼',  '펭귄', '금붕어', '닭'

         '볼펜', '도서관', '박물관',

         '사탕', '과자', '사과', '배', '감', '귤', '포도', '수박', '토마토', '무', '당근', '오이', '배추', '김밥', '빵',
         '초코맛사탕','사과맛사탕','메론맛사탕','포도맛사탕',
         '라면', '음료수', '주스', '우유', '달걀',
         '남학생', '여학생',

         '국어', '영어', '수학', '사회', '과학', '음악', '미술', '체육',

         '오토바이', '트럭', '자동차', '자전거', '비행기', '버스', '배', '기차']

name_dict = {x: n for n, x in enumerate(name_)}
####################################################################
# 한글을 숫자로 바꾸는 preprocessing
a = ['첫[ ]?번째', '두[ ]?번째', '세[ ]?번째', '네[ ]?번째', '다섯[ ]?번째', '여섯[ ]?번째', '일곱[ ]?번째', '여덟[ ]?번째',
     '아홉[ ]?번째', '열[ ]?번째', '스무[ ]?번째', '서른[ ]?번째', '마흔[ ]?번째', '쉰[ ]?번째', '예순[ ]?번째', '일흔[ ]?번째',
     '여든[ ]?번째', '아흔[ ]?번째',

     '첫[ ]?째', '둘[ ]?째', '셋[ ]?째', '넷[ ]?째', '다섯[ ]?째', '여섯[ ]?째', '일곱[ ]?째',
     '여덟[ ]?째', '아홉[ ]?째',

     '일의[ ]?자리', '십의[ ]?자리', '백의[ ]?자리', '천의[ ]?자리',

     '두[ ]?수',

     '한[ ]?개', '두[ ]?개', '세[ ]?개', '네[ ]?개', '다섯[ ]?개', '여섯[ ]?개', '일곱[ ]?개', '여덟[ ]?개', '아홉[ ]?개',
     '열[ ]?개', '스무[ ]?개', '서른[ ]?개', '마흔[ ]?개', '쉰[ ]?개', '예순[ ]?개', '일흔[ ]?개', '여든[ ]?개', '아흔[ ]?개',

     '한[ ]?통', '한[ ]?병',

     '한[ ]?명', '두[ ]?명', '세[ ]?명', '네[ ]?명', '다섯[ ]?명', '여섯[ ]?명', '일곱[ ]?명', '여덟[ ]?명', '아홉[ ]?명',
     '열[ ]?명', '스무[ ]?명', '서른[ ]?명', '마흔[ ]?명', '쉰[ ]?명', '예순[ ]?명', '일흔[ ]?명', '여든[ ]?명', '아흔[ ]?명',

     '한[ ]?가지', '두[ ]?가지', '세[ ]?가지', '네[ ]?가지', '다섯[ ]?가지', '여섯[ ]?가지', '일곱[ ]?가지', '여덟[ ]?가지',
     '아홉[ ]?가지', '열[ ]?가지', '스무[ ]?가지', '서른[ ]?가지', '마흔[ ]?가지', '쉰[ ]?가지', '예순[ ]?가지', '일흔[ ]?가지',
     '여든[ ]?가지', '아흔[ ]?가지',

     '한[ ]?자루', '두[ ]?자루', '세[ ]?자루', '네[ ]?자루', '다섯[ ]?자루', '여섯[ ]?자루', '일곱[ ]?자루', '여덟[ ]?자루',
     '아홉[ ]?자루', '열[ ]?자루', '스무[ ]?자루', '서른[ ]?자루', '마흔[ ]?자루', '쉰[ ]?자루', '예순[ ]?자루', '일흔[ ]?자루',
     '여든[ ]?자루', '아흔[ ]?자루',

     '하나', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉', '열',

     '한[ ]?자리', '두[ ]?자리', '세[ ]?자리', '네[ ]?자리', '다섯[ ]?자리', '여섯[ ]?자리', '일곱[ ]?자리', '여덟[ ]?자리',
     '아홉[ ]?자리', '열[ ]?자리', '스무[ ]?자리', '서른[ ]?자리', '마흔[ ]?자리', '쉰[ ]?자리', '예순[ ]?자리', '일흔[ ]?자리',
     '여든[ ]?자리', '아흔[ ]?자리',

     '한[ ]?마리', '두[ ]?마리', '세[ ]?마리', '네[ ]?마리', '다섯[ ]?마리', '여섯[ ]?마리', '일곱[ ]?마리', '여덟[ ]?마리',
     '아홉[ ]?마리', '열[ ]?마리', '스무[ ]?마리', '서른[ ]?마리', '마흔[ ]?마리', '쉰[ ]?마리', '예순[ ]?마리', '일흔[ ]?마리',
     '여든[ ]?마리', '아흔[ ]?마리',

     '한[ ]?개', '두[ ]?개', '세[ ]?개', '네[ ]?개', '다섯[ ]?개', '여섯[ ]?개', '일곱[ ]?개', '여덟[ ]?개', '아홉[ ]?개',
     '열[ ]?개', '스무[ ]?개', '서른[ ]?개', '마흔[ ]?개', '쉰[ ]?개', '예순[ ]?개', '일흔[ ]?개', '여든[ ]?개', '아흔[ ]?개',

     '한[ ]?명', '두[ ]?명', '세[ ]?명', '네[ ]?명', '다섯[ ]?명', '여섯[ ]?명', '일곱[ ]?명', '여덟[ ]?명', '아홉[ ]?명',
     '열[ ]?명', '스무[ ]?명', '서른[ ]?명', '마흔[ ]?명', '쉰[ ]?명', '예순[ ]?명', '일흔[ ]?명', '여든[ ]?명', '아흔[ ]?명'
     ]

b = ['1번째', '2번째', '3번째', '4번째', '5번째', '6번째', '7번째', '8번째', '9번째',
     '10번째', '20번째', '30번째', '40번째', '50번째', '60번째', '70번째', '80번째', '90번째',

     '1째', '2째', '3째', '4째', '5째', '6째', '7째', '8째', '9째',

     '1자리', '10자리', '100자리', '100자리',

     '2수',

     '1개', '2개', '3개', '4개', '5개', '6개', '7개', '8개', '9개', '10개',
     '20개', '30개', '40개', '50개', '60개', '70개', '80개', '90개',

     '1통', '1병',

     '1명', '2명', '3명', '4명', '5명', '6명', '7명', '8명', '9명', '10명', '20명', '30명', '40명', '50명', '60명',
     '70명', '80명', '90명',
     '1가지', '2가지', '3가지', '4가지', '5가지', '6가지', '7가지', '8가지', '9가지', '10가지', '20가지', '30가지',
     '40가지', '50가지', '60가지', '70가지', '80가지', '90가지',
     '1자루', '2자루', '3자루', '4자루', '5자루', '6자루', '7자루', '8자루', '9자루', '10자루', '20자루', '30자루',
     '40자루', '50자루', '60자루', '70자루', '80자루', '90자루',

     '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',

     '1자리', '2자리', '3자리', '4자리', '5자리', '6자리', '7자리', '8자리', '9자리', '10자리', '20자리', '30자리', '40자리',
     '50자리', '60자리', '70자리', '80자리', '90자리',

     '1마리', '2마리', '3마리', '4마리', '5마리', '6마리', '7마리', '8마리', '9마리', '10마리', '20마리', '30마리', '40마리',
     '50마리', '60마리', '70마리', '80마리', '90마리',

     '1개', '2개', '3개', '4개', '5개', '6개', '7개', '8개', '9개', '10개', '20개', '30개', '40개', '50개', '60개', '70개',
     '80개', '90개',

     '1명', '2명', '3명', '4명', '5명', '6명', '7명', '8명', '9명', '10명', '20명', '30명', '40명', '50명', '60명', '70명',
     '80명', '90명'
     ]

dict_ = {}
for i, j in zip(a, b):
    dict_[i] = j

for i, k in zip(['열', '스물', '서른', '마흔', '쉰', '예순', '일흔', '여든', '아흔'], [10, 20, 30, 40, 50, 60, 70, 80, 90]):
    for j, l in zip(['한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉'], [1, 2, 3, 4, 5, 6, 7, 8, 9]):
        for n in ['번째', '자리', '마리', '개', '명']:
            #             print(i+'[]?'+j+'[]?' +n, '-->',f'{k+l}'+n)
            dict_[i + '[ ]?' + j + '[ ]?' + n] = f'{k + l}' + n


####################################################################
def func1(a):
    """전부다 numberAGC으로 바꾸고 다시 num1~~"""
    a = re.sub('[0-9]+/[0-9]+|[0-9]*\.[0-9]+|[0-9]+', 'numberAGC', a)
    list_ = re.findall('numberAGC', a)
    print("list", list_)
    for n in range(len(list_)):
        a = re.sub('numberAGC', f'number{n}', a, 1)  # 1번씩만 바꿈

    return a


def func2(x):
    """question에서 -+는 빼고 가져와야함.. 안그러면 numbers가 + or -가 되버림"""
    p = re.compile('[0-9]+/[0-9]+|[0-9]*\.[0-9]+|[0-9]+')
    m = p.findall(x)
    string = ''
    for i in m:
        string += i + ' '
    return string


##### 단어 답도 맞추기 위한 작업#####
def func_name(x, name_):
    """근데 이렇게짜면, 같은 이름이더라도 다른 name으로 저장되네"""
    for n in name_:
        x = re.sub(n, '이름대회', x)
    list_ = re.findall('이름대회', x)
    for n in range(len(list_)):
        x = re.sub('이름대회', f'name{n}', x, 1)

    return x


def func_name2(x, name_):
    """문자를 names 정답으로 저장"""
    string = ''
    for i in name_:
        m = re.search(i, x)
        if m is not None:
            string += m.group() + ' '

    return string


def word2number(x, dict_):
    for d in dict_:
        x = re.sub(d, dict_[d], x)
    return x


def eq2num(x, y):
    """
    x : numbers column
    y : equation column
    대신 number가 앞에 붙으면 바꾸지 않는방식
    """
    temp_ = {x: 'number' + str(n) for n, x in enumerate(x.split())}

    for d in temp_:
        #         y = re.sub('[^number]'+d, temp_[d], y)
        y = re.sub(f'[ ]{d}[ ]', f' {temp_[d]} ', y)

    return y


def eq2name(x, y):
    """
    x : name column
    y : equation column
    대신 number가 앞에 붙으면 바꾸지 않는방식

    """
    temp_ = {x: 'name' + str(n) for n, x in enumerate(x.split())}

    for d in temp_:
        y = re.sub(d, temp_[d], y)
    #         y = re.sub(f'[ ]{d}[ ]', f' {temp_[d]} ', y)

    return y


def pp_question(q):
    q = word2number(q, dict_)
    names = func_name2(q, name_)
    nums = func2(q)
    q = func1(q)
    q = func_name(q, name_)
    return {"question": q, "names": names, "nums": nums}

def pp(tp):
    tp['Question'] = tp['Question'].apply(lambda x: word2number(x, dict_))  # 먼저 한글->숫자로 다바꾸고 시작
    tp['Numbers'] = tp['Question'].apply(lambda x: func2(x))  # 숫자 정답 변환
    tp['Names'] = tp['Question'].apply(lambda x: func_name2(x, name_))  # 문자 정답 변환
    # 순서 제대로 지켜야함.
    tp['Question'] = tp['Question'].apply(lambda x: func1(x))  # numbern으로 변환
    tp['Question'] = tp['Question'].apply(lambda x: func_name(x, name_))  # namen으로 변환

    tp['Equation'] = tp['Equation'].apply(lambda x: f' {x} ')  # 양쪽끝에 스페이스 추가해서 num으로 안바뀌는애들 처리
    tp['Equation'] = tp[['Numbers', 'Equation']].apply(lambda x: eq2num(x[0], x[1]), 1)
    tp['Equation'] = tp[['Names', 'Equation']].apply(lambda x: eq2name(x[0], x[1]), 1)

    # 양끝에 스페이스 없애기
    tp['Equation'] = tp['Equation'].apply(lambda x: x.strip())
    tp['Equation'] = tp['Equation'].apply(lambda x: x.strip())

    return tp

def start(path):
    try:
        test = pd.read_csv(path, dtype={'Answer':'str'})
        tp = test.copy()
        print('tye num', tp['Numbers'].dtypes)
        print('tye name', tp['Names'].dtypes)
    except:
        test = pd.read_csv(path.replace('_pp',''))
        #test = test.rename(columns={'question': 'Question', 'equation': 'Equation', 'answer': 'Answer'})
        test = pp(test) # 정제된 dataframe
        test.to_csv(path, index=False)

        tp = test.copy()

        #tp = tp.drop(['Question', 'Equation'], 1)
        #tp = tp.rename(columns={'Question2': 'Question', 'Equation2': 'Equation', 'answer': 'Answer'})


    tp['Numbers'] = tp['Numbers'].astype('str')
    tp['Names'] = tp['Names'].astype('str')

        

    return test, tp



voc1_path  = "./saved_models/vocab1.p"
voc2_path  = "./saved_models/vocab2.p"
saved_path = config.model_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(voc1_path, 'rb') as f:
    voc1 = pickle.load(f, encoding='bytes')
with open(voc2_path, 'rb') as f:
    voc2 = pickle.load(f, encoding='bytes')

model = build_model(config=config, voc1=voc1, voc2=voc2, device=device)
model.load_state_dict(torch.load(saved_path)['state_dict'])
model = model.to(device)
model.eval()


def process_batch(sent1s, voc1, device):
    input_len1 = [len(s) for s in sent1s]
    #     print('input_len1 :',input_len1)
    max_length_1 = max(input_len1)

    sent1s_padded = [pad_seq(s, max_length_1, voc1) for s in sent1s]

    # Convert to [Max_len X Batch]
    sent1_var = torch.LongTensor(sent1s_padded).transpose(0, 1)

    sent1_var = sent1_var.to(device)

    return sent1_var, input_len1


def main_worker(raw_sent):
    pp_output = pp_question(raw_sent)
    ques = [pp_output["question"]]
    names = pp_output["names"].split()
    nums = pp_output["nums"].split()
    sent1s = sents_to_idx(voc1, ques, config.max_length, flag=0)
    sent1_var, _ = process_batch(sent1s, voc1, device)
    decoder_output = model.greedy_decode(ques, sent1_var)
    str_output = "".join(decoder_output[0])
    for i in range(len(names)):
        name = f"name{i}"
        name_val = names[i]
        str_output = str_output.replace(name, name_val)
    for i in range(len(nums)):
        number = f"number{i}"
        number_val = nums[i]
        str_output = str_output.replace(number, number_val)
    return str_output, eval(str_output)

demo = gr.Interface(fn=main_worker, inputs=["text"], outputs=["text", "text"])

demo.launch(share=True)


