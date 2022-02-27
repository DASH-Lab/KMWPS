from prerequisite import *
from model.transformer import *
from utils import *
from dataloader import *

def inference_print(config, voc1_path, voc2_path, saved_path, val_data_path, device):
    with open(voc1_path, 'rb') as f:
        voc1 = pickle.load(f)
    with open(voc2_path, 'rb') as f:
        voc2 = pickle.load(f)

    model = TransformerModel(config, voc1, voc2, device)
    model.load_state_dict(torch.load(saved_path)['state_dict'])
    model = model.to(device)

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

        val_loss, decoder_output = model.greedy_decode(ques, sent1_var, sent2_var, input_len2, criterion,
                                                        validation=True)

        temp_acc_cnt, temp_acc_tot, disp_corr = cal_score(decoder_output, nums, ans, names)
        for n in range(len(decoder_output)):
            str_ = ''
            for i in decoder_output[n]:
                str_ += i

            print(f'pred :{str_}')
            print(f'true : {data["eqn"][n]}')
            print(f'results : {disp_corr[n] == 1}')
            print('')