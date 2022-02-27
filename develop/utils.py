from prerequisite import *

# -- utils --#
def sent_to_idx2(voc, sent, max_length, flag=0):
    if flag == 0:
        idx_vec = []
    else:
        idx_vec = [voc.get_id('<s>')]
    for w in sent.split(' '):
        try:
            idx = voc.get_id(w)
            idx_vec.append(idx)
        except:
            idx_vec.append(voc.get_id('unk'))
    # idx_vec.append(voc.get_id('</s>'))
    if flag == 1 and len(idx_vec) < max_length - 1:
        idx_vec.append(voc.get_id('</s>'))
    return idx_vec


def sent_to_idx(voc, sent, max_length, flag=0):
    if flag == 0:
        idx_vec = []
    else:
        idx_vec = [voc.get_id('<s>')]
    for w in sent.split(' '):
        try:
            idx = voc.get_id(w)
            idx_vec.append(idx)
            idx_space = voc.get_id(' ')
            idx_vec.append(idx_space)
        except:
            idx_vec.append(voc.get_id('unk'))
    # idx_vec.append(voc.get_id('</s>'))
    if flag == 1 and len(idx_vec) < max_length - 1:
        idx_vec.append(voc.get_id('</s>'))
    return idx_vec


def sents_to_idx(voc, sents, max_length, flag=0):
    all_indexes = []
    for sent in sents:
        all_indexes.append(sent_to_idx(voc, sent, max_length, flag))
    return all_indexes


def sent_to_tensor(voc, sentence, device, max_length):
    indexes = sent_to_idx(voc, sentence, max_length)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def batch_to_tensor(voc, sents, device, max_length):
    batch_sent = []
    # batch_label = []
    for sent in sents:
        sent_id = sent_to_tensor(voc, sent, device, max_length)
        batch_sent.append(sent_id)

    return batch_sent


def idx_to_sent(voc, tensor, no_eos=False):
    sent_word_list = []
    for idx in tensor:
        word = voc.get_word(idx.item())
        if no_eos:
            if word != '</s>':
                sent_word_list.append(word)
            # else:
            # 	break
        else:
            sent_word_list.append(word)
    return sent_word_list


def idx_to_sents(voc, tensors, no_eos=False):
    tensors = tensors.transpose(0, 1)
    batch_word_list = []
    for tensor in tensors:
        batch_word_list.append(idx_to_sent(voc, tensor, no_eos))

    return batch_word_list


def pad_seq(seq, max_length, voc):
    seq += [voc.get_id('</s>') for i in range(max_length - len(seq))]
    return seq


def sort_by_len(seqs, input_len, device=None, dim=1):
    orig_idx = list(range(seqs.size(dim)))

    # Index by which sorting needs to be done
    sorted_idx = sorted(orig_idx, key=lambda k: input_len[k], reverse=True)
    sorted_idx = torch.LongTensor(sorted_idx)
    if device:
        sorted_idx = sorted_idx.to(device)

    sorted_seqs = seqs.index_select(1, sorted_idx)
    sorted_lens = [input_len[i] for i in sorted_idx]

    # For restoring original order
    orig_idx = sorted(orig_idx, key=lambda k: sorted_idx[k])
    orig_idx = torch.LongTensor(orig_idx)
    if device:
        orig_idx = orig_idx.to(device)
    return sorted_seqs, sorted_lens, orig_idx


def restore_order(seqs, input_len, orig_idx):
    orig_seqs = [seqs[i] for i in orig_idx]
    orig_lens = [input_len[i] for i in orig_idx]
    return orig_seqs, orig_lens


def process_batch(sent1s, sent2s, voc1, voc2, device):
    input_len1 = [len(s) for s in sent1s]
    input_len2 = [len(s) for s in sent2s]
    #     print('input_len1 :',input_len1)
    max_length_1 = max(input_len1)
    max_length_2 = max(input_len2)

    sent1s_padded = [pad_seq(s, max_length_1, voc1) for s in sent1s]
    sent2s_padded = [pad_seq(s, max_length_2, voc2) for s in sent2s]

    # Convert to [Max_len X Batch]
    sent1_var = Variable(torch.LongTensor(sent1s_padded)).transpose(0, 1)
    sent2_var = Variable(torch.LongTensor(sent2s_padded)).transpose(0, 1)

    sent1_var = sent1_var.to(device)
    sent2_var = sent2_var.to(device)
    #     print('sent1_var :',sent1_var.shape)

    return sent1_var, sent2_var, input_len1, input_len2


#############################################
def cal_score2(outputs, nums, ans):
    """
    덧셈이 없으면 16+16 --> 161616 이런식이 될수있음
    """
    corr = 0
    tot = len(nums)
    disp_corr = []
    for i in range(len(outputs)):
        op = outputs[i]
        num = nums[i].split()
        # num = [float(nu) for nu in num]
        answer = ans[i].item()
        str_ = ''
        for o_ in op:
            str_ += o_
        op = str_
        for n, j in enumerate(num):
            op = re.sub(f'number{n}', j, op)
        try:
            try:  # round로 내보내기
                pred = round(exec(op), 2)
            except ValueError:  # string이면 그대로 내보내기
                pred = exec(op)
        except SyntaxError:  # 셈식이 안맞는경우
            pred = -999
        except NameError:  # 셈식이 안맞는경우
            pred = -999
        except ZeroDivisionError:
            pred = -999

        if abs(pred - answer) <= 0.01:
            corr += 1
            #             tot+=1
            disp_corr.append(1)
        else:
            #             tot+=1
            disp_corr.append(0)

    return corr, tot, disp_corr


def cal_score3(outputs, nums, ans):
    """
    아예 string으로 비교를 하자
    """
    corr = 0
    tot = len(nums)
    disp_corr = []
    for i in range(len(outputs)):
        op = outputs[i]
        num = nums[i].split()
        # num = [float(nu) for nu in num]
        answer = ans[i].item()
        str_ = ''
        for o_ in op:
            str_ += o_
        op = str_
        for n, j in enumerate(num):
            op = re.sub(f'number{n}', j, op)
        try:
            try:  # round로 내보내기 # '3+1' 을 바로 exec하면 아무값도 return안됨.. print를 해야함

                round(eval(op), 2)  # temp, for verify
                op2 = round(float(eval(op)), 2)  # "print(" + round(eval(op), 2) + ")"
                pred = str(op2)  # print값은 return이 안됨...

            except ValueError:  # string이면 그대로 내보내기(factorial, comb인 경우가 있음)
                try:  # 여기서 또 3C-20 이런경우가 생김..
                    pred = exec(op)
                except ValueError:  # k must be a non-negative integer
                    pred = str('no Answer1')

            except TypeError:  # nontype일 경우가 있네
                pred = str('no Answer2')

        except SyntaxError:  # 셈식이 안맞는경우(3+)
            pred = str('no Answer3')
        except NameError:  # 셈식이 안맞는경우(name5 not defined)
            pred = str('no Answer4')
        except ZeroDivisionError:
            pred = str('no Answer5')

        # answer
        try:
            answer = str(round(answer, 2))
        except ValueError:
            pass

        #         if
        if pred == answer:
            corr += 1
            #             tot+=1
            disp_corr.append(1)
        else:
            #             tot+=1
            disp_corr.append(0)

    return corr, tot, disp_corr


def cal_score(outputs, nums, ans, names):
    """
    아예 string으로 비교를 하자
    """
    corr = 0
    tot = len(nums)
    disp_corr = []
    for i in range(len(outputs)):
        op = outputs[i]
        num = nums[i].split()
        name = names[i]
        # num = [float(nu) for nu in num]
        answer = ans[i]  # number일 경우는 .item()을 해야함
        str_ = ''
        for o_ in op:  # op : 'number0 + number1'
            str_ += o_
        op = str_
        for n, j in enumerate(num):
            op = re.sub(f'number{n}', j, op)
        for n, j in enumerate(name):
            op = re.sub(f'name{n}', j, op)

        # try:
        try:  # round로 내보내기 # '3+1' 을 바로 exec하면 아무값도 return안됨.. print를 해야함
            # 만약, op가 string이라면 다음으로 넘어갈것임
            op2 = round(float(eval(op)), 2)
            pred = str(op2)  # print값은 return이 안됨...

        except ValueError:  # string이면 그대로 내보내기(factorial, comb인 경우가 있음)
            try:  # 여기서 또 3C-20 이런경우가 생김..
                pred = eval(op)
            except:  # k must be a non-negative integer
                pred = str('no Answer')

        except:  # TypeError : # nontype일 경우가 있네
            pred = str('no Answer')

        #         except IndexError: # 내생각에 인덱싱하는 코드를 예측하는데 그부분에서 나는 에러같음
        #             pred = str('no Answer')

        #         except AttributeError: # int를 indexing하려니깐,,
        #             pred = str('no Answer')

        #         except SyntaxError: # 셈식이 안맞는경우(3+)
        #             pred = str('no Answer')
        #         except NameError: # 셈식이 안맞는경우(name5 not defined)
        #             pred = str('no Answer')
        #         except ZeroDivisionError:
        #             pred = str('no Answer')
        #         except OverflowError:
        #             pred = str('no Answer')

        # answer
        try:
            #             print('just answer:',answer)
            answer = str(round(float(eval(answer)), 2))
        except ValueError:
            pass
        except NameError:
            pass
        except TypeError:  # F 같은걸 eval하면, 함수 F를 불러오게됨..
            pass

        #         print('answer score: ',answer)
        #         print('predict score:', pred)
        if pred == answer:
            corr += 1
            #             tot+=1
            disp_corr.append(1)
        else:
            #             tot+=1
            disp_corr.append(0)

    return corr, tot, disp_corr


"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        if ratio > 1E-1:
            bp = math.exp(1 - 1. / ratio)
        else:
            bp = 1E-2

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


# train, val
"""bleau일단 제거상태// train, val둘다 만져야함"""


# global log_folder
# global model_folder
# global result_folder
# global data_path
# global board_path

def get_scheduler(optimizer, config):
    if config.scheduler == 'ReduceLROnPlateau':
        scheduler = None
    #         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience,
    #                                       min_lr = 1e-5, verbose=True, eps=args.eps)
    elif config.scheduler == 'CosineAnnealingLR':
        print('scheduler : Cosineannealinglr')
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=1e-6, last_epoch=-1)
    #     elif args.scheduler=='CosineAnnealingWarmRestarts':
    #         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr, last_epoch=-1)
    #     elif args.scheduler == 'MultiStepLR':
    #         scheduler = MultiStepLR(optimizer, milestones=args.decay_epoch, gamma= args.factor, verbose=True)
    #     elif args.scheduler == 'OneCycleLR':
    #         scheduler = OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
    #                                       max_lr=1e-3, epochs=args.epochs, steps_per_epoch=len(train_loader))
    else:
        scheduler = None
        print('scheduler is None')
    return scheduler


def get_optimizer(model, config):
    my_list = ['embedding1']
    embed_params = list(map(lambda x: x[1], list(filter(lambda kv: my_list[0] in kv[0], model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: my_list[0] not in kv[0], model.named_parameters()))))

    # optimizer
    if config.opt == 'adam':
        optimizer = optim.Adam(
            [{"params": embed_params, "lr": config.emb_lr},
             {"params": base_params, "lr": config.lr}]
        )
    elif config.opt == 'adamw':
        optimizer = optim.AdamW(
            [{"params": embed_params, "lr": config.emb_lr},
             {"params": base_params, "lr": config.lr}]
        )
    elif config.opt == 'adadelta':
        optimizer = optim.Adadelta(
            [{"params": embed_params, "lr": config.emb_lr},
             {"params": base_params, "lr": config.lr}]
        )
    elif config.opt == 'asgd':
        optimizer = optim.ASGD(
            [{"params": embed_params, "lr": config.emb_lr},
             {"params": base_params, "lr": config.lr}]
        )
    elif config.opt == 'sgd':
        optimizer = optim.SGD(
            [{"params": embed_params, "lr": config.emb_lr},
             {"params": base_params, "lr": config.lr}]
        )
    return optimizer