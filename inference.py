from prerequisite import *
from model.tiny_transformer import *
from utils import *
from dataloader import *
from train import build_model
# import pickle5 as pickle
import pickle
import argparse

parser = argparse.ArgumentParser(description='Script of Brid call')
parser.add_argument('--gpu', '-gpu', type=str, default='0', help='gpu')
parser.add_argument('--hidden_layer', '-hl', type=int, default=12)
parser.add_argument('--hidden_size', '-hs', type=int, default=768)
parser.add_argument('--intermediate_size', '-is', type=int, default=3072)
parser.add_argument('--distill', action='store_true', help='distill [True/False]')
parser.add_argument('--teacher_path', '-tp', type=str, default='./models/bert12/n12_h768_i_3072_kobert.pth', help='teacher model path')

parser.add_argument('--num_decoder_layer', '-ndl', type=int, default=1)
parser.add_argument('--decoder_head', '-dh', type=int, default=12)

parser.add_argument('--model_path', '-mp', type=str, default=None, help='model path')


args = parser.parse_args()
class config:
    # ---- factor ---- #
    debug = False
    full_cv = False
    mode = 'train'
    gpu = args.gpu
    dropout = 0.1
    # -- transformer decoder -- #
    heads = args.decoder_head  # 4
    encoder_layers = args.num_decoder_layer
    decoder_layers = args.num_decoder_layer
    d_model = 768
    d_ff = 1024  # 256
    # -- #
    batch_size =64 # 16
    embedding = 'bert' # ['bert','roberta']
    emb_name = 'monologg/kobert'#'bert-base-uncased' #'monologg/distilkobert'#'HanBert-54kN-torch'#'skt/kobert-base-v1'#'monologg/kobigbird-bert-base'  # 'bert-base-uncased' # ['bert-base-uncased', 'roberta-base','roberta-large]
    mawps_vocab = True

    max_length = 30  # 30
    vocab_size = 30000  # 30000
    init_range = 0.08  # 'Initialization range for seq2seq model'
    max_grad_norm = 0.25
    # - save - #
    save_model = False
    model_path = f'./saved_models/'
    ckpt = 'bert_good'  # model name

    # -- else --#
    val_outputs = False  # Show full validation outputs
    freeze_emb = True
    interval   = 1      # evaluation interval epoch
    # -- encoder -- #
    num_hidden_layers = args.hidden_layer 
    hidden_size       = args.hidden_size #768#252
    intermediate_size = args.intermediate_size #3072#786
    model_pth_name = args.model_path

    # ---- Else ---- #
    num_workers = 8
    seed = 92

def inference_print(config, voc1_path, voc2_path, saved_path, device):
    with open(voc1_path, 'rb') as f:
        voc1 = pickle.load(f, encoding='bytes')
    with open(voc2_path, 'rb') as f:
        voc2 = pickle.load(f, encoding='bytes')

    model = build_model(config=config, voc1=voc1, voc2=voc2, device=device)
    if saved_path != 'None':
        model.load_state_dict(torch.load(saved_path)['state_dict'])
    model = model.to(device)

    print('model loaded')
    model.eval()

    val_dataloader_main = load_data(config)[1]
    criterion = nn.CrossEntropyLoss()
    df = pd.DataFrame()
    preds, true, correct = [],[],[]
    val_acc_epoch_cnt = 0.0
    val_acc_epoch_tot = 0.0
    for data in val_dataloader_main:
        sent1s = sents_to_idx(voc1, data['ques'], config.max_length, flag=0)
        sent2s = sents_to_idx(voc2, data['eqn'], config.max_length, flag=0)
        nums = data['nums']
        names = data['names']
        ans = data['ans']
        ques = data['ques']

        sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

        val_loss, decoder_output = model.greedy_decode(ques, sent1_var, sent2_var, input_len2, criterion, validation=True)

        temp_acc_cnt, temp_acc_tot, disp_corr = cal_score(decoder_output, nums, ans, names)
        val_acc_epoch_cnt += temp_acc_cnt
        val_acc_epoch_tot += temp_acc_tot

        for n in range(len(decoder_output)):
            str_ = ''
            for i in decoder_output[n]:
                str_ += i

            # print(f'pred :{str_}') ; preds.append(str_)
            # print(f'true : {data["eqn"][n]}') ; true.append(data['eqn'][n])
            # print(f'results : {disp_corr[n] == 1}') ; correct.append(disp_corr[n] == 1)
            # print('')

    df['preds'] = preds
    df['true'] = true
    df['correct'] = correct

    #df.to_csv('/home/leegwang/project/KMWPS/develop/models/temp.csv', index=False)
    print('acc:',val_acc_epoch_cnt / val_acc_epoch_tot)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time
prev = time.time()
inference_print(config(),
                "./saved_models/vocab1.p",
                "./saved_models/vocab2.p",
                f'{config().model_pth_name}',
                device)
print(f"time: {time.time() - prev}")
