from prerequisite import *
from model.transformer import *
from utils import *
from dataloader import *
from train import build_model
import pickle5 as pickle


class config:
    # ---- factor ---- #
    debug = False
    full_cv = False
    mode = 'train'
    gpu = "2"
    dropout = 0.1
    heads = 8  # 4
    encoder_layers = 1
    decoder_layers = 1
    d_model = 768
    d_ff = 1024  # 256
    lr = 1e-4
    emb_lr = 1e-5
    batch_size = 1
    epochs = 500
    embedding = 'bert' # ['bert','roberta']
    emb_name = 'bert-base-uncased' #'monologg/distilkobert'#'HanBert-54kN-torch'#'skt/kobert-base-v1'#'monologg/kobigbird-bert-base'  # 'bert-base-uncased' # ['bert-base-uncased', 'roberta-base']
    mawps_vocab = True

    max_length = 100  # 30
    vocab_size = 30000  # 30000
    init_range = 0.08  # 'Initialization range for seq2seq model'
    max_grad_norm = 0.25
    opt = 'adamw'  # choices=['adam', 'adamw', 'adadelta', 'sgd', 'asgd']
    scheduler = None  # 'CosineAnnealingLR'#None#'CosineAnnealingLR'
    T_max = 50

    early_stopping = 500
    # - save - #
    save_model = False
    model_path = f'./saved_models/'
    ckpt = 'bert_good'  # model name

    # -- else --#
    val_outputs = False  # Show full validation outputs
    freeze_emb = False
    # ---- Else ---- #
    num_workers = 8
    seed = 92

def inference_print(config, voc1_path, voc2_path, saved_path, device):
    with open(voc1_path, 'rb') as f:
        voc1 = pickle.load(f, encoding='bytes')
    with open(voc2_path, 'rb') as f:
        voc2 = pickle.load(f, encoding='bytes')

    model = build_model(config=config, voc1=voc1, voc2=voc2, device=device)
    #model.load_state_dict(torch.load(saved_path)['state_dict'])
    model = model.to(device)

    print('model loaded')
    model.eval()

    val_dataloader_main = load_data(config)[1]
    criterion = nn.CrossEntropyLoss()
    
    for data in val_dataloader_main:
        sent1s = sents_to_idx(voc1, data['ques'], config.max_length, flag=0)
        sent2s = sents_to_idx(voc2, data['eqn'], config.max_length, flag=0)
        nums = data['nums']
        names = data['names']
        ans = data['ans']
        ques = data['ques']

        sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

        val_loss, decoder_output = model.greedy_decode(ques, sent1_var, sent2_var, input_len2, criterion, validation=True)

        # temp_acc_cnt, temp_acc_tot, disp_corr = cal_score(decoder_output, nums, ans, names)
        # for n in range(len(decoder_output)):
        #     str_ = ''
        #     for i in decoder_output[n]:
        #         str_ += i

        #     print(f'pred :{str_}')
        #     print(f'true : {data["eqn"][n]}')
        #     print(f'results : {disp_corr[n] == 1}')
        #     print('')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time
prev = time.time()
inference_print(config(),
                "/home/taejune/KMWPS/model/vocab1.p",
                "/home/taejune/KMWPS/model/vocab2.p",
                "/home/taejune/KMWPS/models/best_models_BERT.pth",
                device)
print(f"time: {time.time() - prev}")
