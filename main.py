from prerequisite import *
from model.transformer import *
from utils import *
from dataloader import *
from preprocess import *
from vocab import *
from train import *


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
    batch_size = 32
    epochs = 500
    embedding = 'roberta' # ['bert','roberta']
    emb_name = 'roberta-large' #'monologg/distilkobert'#'HanBert-54kN-torch'#'skt/kobert-base-v1'#'monologg/kobigbird-bert-base'  # 'bert-base-uncased' # ['bert-base-uncased', 'roberta-base']
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


data_dir = './data/'
#os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
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

    if config.full_cv:
        global data_path
        data_name = config.dataset
        data_path = data_path + data_name + '/'

        fold_acc_score = 0.0
        folds_scores = []
        for z in range(5):
            run_name = config.run_name + '_fold' + str(z)
            config.dataset = 'fold' + str(z)
            config.log_path = os.path.join(log_folder, run_name)
            config.model_path = os.path.join(model_folder, run_name)
            config.board_path = os.path.join(board_path, run_name)
            config.outputs_path = os.path.join(outputs_folder, run_name)

            vocab1_path = os.path.join(config.model_path, 'vocab1.p')
            vocab2_path = os.path.join(config.model_path, 'vocab2.p')
            config_file = os.path.join(config.model_path, 'config.p')
            log_file = os.path.join(config.log_path, 'log.txt')

            if config.results:
                config.result_path = os.path.join(result_folder,
                                                  'val_results_{}_{}.json'.format(data_name, config.dataset))

            if is_train:
                create_save_directories(config.log_path)
                create_save_directories(config.model_path)
                create_save_directories(config.outputs_path)
            else:
                create_save_directories(config.log_path)
                create_save_directories(config.result_path)

            logger = get_logger(run_name, log_file, logging.DEBUG)
            writer = SummaryWriter(config.board_path)

            #             logger.debug('Created Relevant Directories')
            logger.info('Experiment Name: {}'.format(config.run_name))

            '''Read Files and create/load Vocab'''
            if is_train:
                train_dataloader, val_dataloader = load_data(config, logger)

                #                 logger.debug('Creating Vocab...')

                voc1 = Voc1()
                voc1.create_vocab_dict(config, train_dataloader)

                # Removed
                # voc1.add_to_vocab_dict(config, val_dataloader)

                voc2 = Voc2(config)
                voc2.create_vocab_dict(config, train_dataloader)

                # Removed
                # voc2.add_to_vocab_dict(config, val_dataloader)

                logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

                with open(vocab1_path, 'wb') as f:
                    pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(vocab2_path, 'wb') as f:
                    pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info('Vocab saved at {}'.format(vocab1_path))

            else:
                test_dataloader = load_data(config, logger)
                logger.info('Loading Vocab File...')

                with open(vocab1_path, 'rb') as f:
                    voc1 = pickle.load(f)
                with open(vocab2_path, 'rb') as f:
                    voc2 = pickle.load(f)

                logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

            # TO DO : Load Existing Checkpoints here
            checkpoint = get_latest_checkpoint(config.model_path, logger)

            if is_train:
                model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

                logger.info('Initialized Model')

                if checkpoint == None:
                    min_val_loss = torch.tensor(float('inf')).item()
                    min_train_loss = torch.tensor(float('inf')).item()
                    max_val_bleu = 0.0
                    max_val_acc = 0.0
                    max_train_acc = 0.0
                    best_epoch = 0
                    epoch_offset = 0
                else:
                    epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                        load_checkpoint(model, config.mode, checkpoint, logger, device)

                max_val_acc = train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger,
                                          epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss,
                                          max_train_acc, best_epoch, writer)

            else:
                gpu = config.gpu
                mode = config.mode
                dataset = config.dataset
                batch_size = config.batch_size
                with open(config_file, 'rb') as f:
                    config = AttrDict(pickle.load(f))
                    config.gpu = gpu
                    config.mode = mode
                    config.dataset = dataset
                    config.batch_size = batch_size

                with open(config_file, 'rb') as f:
                    config = AttrDict(pickle.load(f))
                    config.gpu = gpu

                model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

                epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                    load_checkpoint(model, config.mode, checkpoint, logger, device)

                logger.info('Prediction from')
                od = OrderedDict()
                od['epoch'] = ep_offset
                od['min_train_loss'] = min_train_loss
                od['min_val_loss'] = min_val_loss
                od['max_train_acc'] = max_train_acc
                od['max_val_acc'] = max_val_acc
                od['max_val_bleu'] = max_val_bleu
                od['best_epoch'] = best_epoch
                print_log(logger, od)

                test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
                logger.info('Accuracy: {}'.format(test_acc_epoch))

            fold_acc_score += max_val_acc
            folds_scores.append(max_val_acc)

        fold_acc_score = fold_acc_score / 5
        store_val_results(config, fold_acc_score, folds_scores)
        logger.info('Final Val score: {}'.format(fold_acc_score))

    else:
        '''Run Config files/paths'''
        #         config.log_path = log_folder
        config.model_path = model_folder
        #         config.board_path = board_path
        #         config.outputs_path = outputs_folder

        vocab1_path = os.path.join(config.model_path, 'vocab1.p')
        vocab2_path = os.path.join(config.model_path, 'vocab2.p')
        config_file = os.path.join(config.model_path, 'config.p')
        #         log_file = os.path.join(config.log_path, 'log.txt')

        if is_train:
            create_save_directories(config.model_path) # model_path가 없으면 디렉토리를 만듦

        '''Read Files and create/load Vocab'''
        if is_train:
            train_dataloader, val_dataloader = load_data(config)

            voc1 = Voc1()
            voc1.create_vocab_dict(config, train_dataloader)

            voc2 = Voc2(config)
            voc2.create_vocab_dict(config, train_dataloader)

            with open(vocab1_path, 'wb') as f:
                pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(vocab2_path, 'wb') as f:
                pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)
        #           print(voc2.id2w)
        #           print(voc2.w2c)

        else:  # is test
            test_dataloader = load_data(config, logger)
            logger.info('Loading Vocab File...')

            with open(vocab1_path, 'rb') as f:
                voc1 = pickle.load(f)
            with open(vocab2_path, 'rb') as f:
                voc2 = pickle.load(f)

        if is_train:
            model = build_model(config=config, voc1=voc1, voc2=voc2, device=device)

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
            train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config,
                        epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc,
                        best_epoch)

        else:  # if not training
            gpu = config.gpu
            mode = config.mode
            dataset = config.dataset
            batch_size = config.batch_size
            with open(config_file, 'rb') as f:
                config = AttrDict(pickle.load(f))
                config.gpu = gpu
                config.mode = mode
                config.dataset = dataset
                config.batch_size = batch_size

            model = build_model(config=config, voc1=voc1, voc2=voc2, device=device)

            epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
                load_checkpoint(model, config.mode, checkpoint, logger, device)

            print('Prediction from')
            od = OrderedDict()
            od['epoch'] = ep_offset
            od['min_train_loss'] = min_train_loss
            od['min_val_loss'] = min_val_loss
            od['max_train_acc'] = max_train_acc
            od['max_val_acc'] = max_val_acc
            od['max_val_bleu'] = max_val_bleu
            od['best_epoch'] = best_epoch

            test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
            print('Accuracy: {}'.format(test_acc_epoch))


model_folder = 'models'
data_path = './'

main(config)