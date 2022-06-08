from prerequisite import *
from model.tiny_transformer import *
from utils import *
from dataloader import *
from preprocess import *
from vocab import *
from train import *
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

parser.add_argument('--exp', '-exp', type=str, default='', help='experiments name')


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
    lr = 1e-4
    emb_lr = 1e-5
    batch_size =64 # 16
    epochs = 500
    embedding = 'bert' # ['bert','roberta']
    emb_name = 'monologg/kobert'#'bert-base-uncased' #'monologg/distilkobert'#'HanBert-54kN-torch'#'skt/kobert-base-v1'#'monologg/kobigbird-bert-base'  # 'bert-base-uncased' # ['bert-base-uncased', 'roberta-base','roberta-large]
    mawps_vocab = True

    max_length = 30  # 30
    vocab_size = 30000  # 30000
    init_range = 0.08  # 'Initialization range for seq2seq model'
    max_grad_norm = 0.25
    opt = 'adamw'  # choices=['adam', 'adamw', 'adadelta', 'sgd', 'asgd']
    scheduler = None  # 'CosineAnnealingLR'#None#'CosineAnnealingLR'
    T_max = epochs

    early_stopping = 500
    # - save - #
    save_model = False
    model_path = f'./saved_models/'
    ckpt = 'bert_good'  # model name

    # -- else --#
    val_outputs = False  # Show full validation outputs
    freeze_emb = True
    interval   = 1      # evaluation interval epoch
    exp        = args.exp # experiments name
    # -- encoder -- #
    num_hidden_layers = args.hidden_layer 
    hidden_size       = args.hidden_size #768#252
    intermediate_size = args.intermediate_size #3072#786
    model_pth_name = f'n{num_hidden_layers}_h{hidden_size}_i_{intermediate_size}_head_{heads}_{emb_name.split("/")[1]}_exp-{exp}.pth'

    # -- related distill -- #
    distill = args.distill
    teacher_path = args.teacher_path
    temperature = 0.5

    # ---- Else ---- #
    num_workers = 8
    seed = 92


data_dir = './data/'
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic


import random

set_seeds(seed=config.seed)


def main(config):
    '''read arguments'''
    mode = config.mode
    if mode == 'train':
        is_train = True
    else:
        is_train = False

    ''' Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    '''GPU initialization'''
    device = gpu_init_pytorch(config.gpu)
    print("device:", device)

    '''Run Config files/paths'''
    #         config.log_path = log_folder
    config.model_path = model_folder
    #         config.board_path = board_path
    #         config.outputs_path = outputs_folder

    vocab1_path = os.path.join(config.model_path, 'vocab1.p')
    vocab2_path = os.path.join(config.model_path, 'vocab2.p')
    config_file = os.path.join(config.model_path, 'config.p')
    #         log_file = os.path.join(config.log_path, 'log.txt')

    create_save_directories(config.model_path) # model_path가 없으면 디렉토리를 만듦

    '''Read Files and create/load Vocab'''
    train_dataloader, val_dataloader = load_data(config)

    voc1 = Voc1()
    voc1.create_vocab_dict(config, train_dataloader)

    voc2 = Voc2(config)
    voc2.create_vocab_dict(config, train_dataloader)

    with open(vocab1_path, 'wb') as f:
        pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(vocab2_path, 'wb') as f:
        pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)


    if config.distill:
        teacher_model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, teacher=True)
        teacher_model.load_state_dict(torch.load(config.teacher_path)['state_dict'])
    else:
        teacher_model = None
    student_model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, teacher=False)

    print('Initialized Model')

    # checkpoint is none
    min_val_loss = torch.tensor(float('inf')).item()
    min_train_loss = torch.tensor(float('inf')).item()
    max_val_bleu = 0.0
    max_val_acc = 0.0
    max_train_acc = 0.0
    best_epoch = 0
    epoch_offset = 0

    # training
    train_model(teacher_model,student_model, train_dataloader, val_dataloader, voc1, voc2, device, config,
                epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc,
                best_epoch)



model_folder = 'models'
data_path = './'

main(config)