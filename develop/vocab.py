from prerequisite import *
# vocab & useful
def gpu_init_pytorch(gpu_num):
    '''
        Initialize GPU

        Args:
            gpu_num (int): Which GPU to use
        Returns:
            device (torch.device): GPU device
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(device)
    # 태준수정
    #device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    return device


def create_save_directories(path):
    if not os.path.exists(path):
        print("notice: directory doesn't exist => Making Directory...")
        os.makedirs(path)


def save_checkpoint(state, epoch, model_path, ckpt):
    '''
        Saves the model state along with epoch number. The name format is important for
        the load functions. Don't mess with it.

        Args:
            state (dict): model state
            epoch (int): current epoch
            model_path (string): directory to save model
            ckpt (string): checkpoint name
    '''

    ckpt_path = os.path.join(model_path, '{}_{}.pt'.format(ckpt, epoch))
    print('Saving Checkpoint at : {}'.format(ckpt_path))
    torch.save(state, ckpt_path)


class Voc1:
    def __init__(self):
        self.trimmed = False
        self.frequented = False
        self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
        self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
        self.w2c = {}
        self.nwords = 3

    def add_word(self, word):
        if word not in self.w2id:
            self.w2id[word] = self.nwords
            self.id2w[self.nwords] = word
            self.w2c[word] = 1
            self.nwords += 1
        else:
            self.w2c[word] += 1

    def add_sent(self, sent):
        for word in sent.split():
            self.add_word(word)

    def most_frequent(self, topk):
        # if self.frequented == True:
        # 	return
        # self.frequented = True

        keep_words = []
        count = 3
        sort_by_value = sorted(
            self.w2c.items(), key=lambda kv: kv[1], reverse=True)
        for word, freq in sort_by_value:
            keep_words += [word] * freq
            count += 1
            if count == topk:
                break

        self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
        self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
        self.w2c = {}
        self.nwords = 3

        for word in keep_words:
            self.add_word(word)

    def trim(self, mincount):
        if self.trimmed == True:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.w2c.items():
            if v >= mincount:
                keep_words += [k] * v

        self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
        self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
        self.w2c = {}
        self.nwords = 3
        for word in keep_words:
            self.addWord(word)

    def get_id(self, idx):
        return self.w2id[idx]

    def get_word(self, idx):
        return self.id2w[idx]

    def create_vocab_dict(self, args, train_dataloader):
        for data in train_dataloader:
            for sent in data['ques']:
                self.add_sent(sent)

        self.most_frequent(args.vocab_size)
        assert len(self.w2id) == self.nwords
        assert len(self.id2w) == self.nwords

    def add_to_vocab_dict(self, args, dataloader):
        for data in dataloader:
            for sent in data['ques']:
                self.add_sent(sent)

        self.most_frequent(args.vocab_size)
        assert len(self.w2id) == self.nwords
        assert len(self.id2w) == self.nwords


class Voc2:
    def __init__(self, config):
        self.frequented = False
        if config.mawps_vocab:
            # '0.25', '8.0', '0.05', '60.0', '7.0', '5.0', '2.0', '4.0', '1.0', '12.0', '100.0', '25.0', '0.1', '3.0', '0.01', '0.5', '10.0'
            self.w2id = {'<s>': 0, '</s>': 1, '+': 2, '-': 3, '*': 4, '/': 5, 'number0': 6, 'number1': 7, 'number2': 8,
                         'number3': 9, 'number4': 10, 'number5': 11, 'number6': 12, 'number7': 13, 'number8': 14,
                         'number9': 15, 'number10': 16, '(': 17, ')': 18, 'unk': 19, 'math.comb(': 20,
                         'math.factorial(': 21,
                         'max([': 22, 'min([': 23, '])': 24, ' ': 25}

            self.id2w = {0: '<s>', 1: '</s>', 2: '+', 3: '-', 4: '*', 5: '/', 6: 'number0', 7: 'number1', 8: 'number2',
                         9: 'number3', 10: 'number4', 11: 'number5', 12: 'number6', 13: 'number7',
                         14: 'number8', 15: 'number9', 16: 'number10', 17: '(', 18: ')', 19: 'unk', 20: 'math.comb',
                         21: 'math.factorial', 22: 'max([', 23: 'min([', 24: '])', 25: ' '}

            self.w2c = {'+': 0, '-': 0, '*': 0, '/': 0, 'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0,
                        'number4': 0, 'number5': 0, 'number6': 0, 'number7': 0, 'number8': 0, 'number9': 0
                , 'number10': 0, '(': 0, ')': 0, 'unk': 0, 'math.comb(': 0, 'math.factorial(': 0,
                        'max([': 0, 'min([': 0, '])': 0, ' ': 0}
            self.nwords = 26

    def add_word(self, word):
        if word not in self.w2id:  # IT SHOULD NEVER GO HERE!!
            self.w2id[word] = self.nwords
            self.id2w[self.nwords] = word
            self.w2c[word] = 1
            self.nwords += 1
        else:
            self.w2c[word] += 1

    def add_sent(self, sent):
        for word in sent.split():
            self.add_word(word)
        # elf.add_word(sent)

    def get_id(self, idx):
        return self.w2id[idx]

    def get_word(self, idx):
        return self.id2w[idx]

    def create_vocab_dict(self, args, train_dataloader):
        for data in train_dataloader:
            for sent in data['eqn']:
                self.add_sent(sent)

        assert len(self.w2id) == self.nwords
        assert len(self.id2w) == self.nwords

    def add_to_vocab_dict(self, args, dataloader):
        for data in dataloader:
            for sent in data['eqn']:
                self.add_sent(sent)

        assert len(self.w2id) == self.nwords
        assert len(self.id2w) == self.nwords


class Voc23:
    def __init__(self, config):
        self.frequented = False
        if config.mawps_vocab:
            self.w2id = {'<s>': 0, '</s>': 1, '+': 2, '-': 3, '*': 4, '/': 5, 'number0': 6, 'number1': 7, 'number2': 8,
                         'number3': 9, 'number4': 10, 'number5': 11, 'number6': 12, 'number7': 13, 'number8': 14,
                         'number9': 15, 'number10': 16, '(': 17, ')': 18, 'unk': 19, 'math.comb(': 20,
                         'math.factorial(': 21,
                         'max': 22, 'min': 23, '[': 24, ']': 25, '//': 26, '%': 27, '<': 28, '>': 29, '>=': 30,
                         '<=': 31, '==': 32,
                         '!=': 33, '**': 34, 'if': 35, 'else': 36, 'elif': 37, 'sum': 38, 'bool': 39, 'not': 40,
                         'and': 41,
                         'or': 42, 'abs': 43, 'arr': 44, 'printx for x in range': 45, 'for i in': 46,
                         'for x in range': 47,
                         'for n in range': 48, 'for index in range': 49, '민영': 50, '유나': 51, '정국': 52, '유정': 53,
                         '태형': 54, '남준': 55, '윤기': 56, '호석': 57, '지민': 58, '석진': 59, '은지': 60, 'A': 61,
                         'B': 62, 'C': 63, 'D': 64, 'E': 65, 'F': 66, 'G': 67, 'H': 68, 'I': 69, 'J': 70, 'K': 71,
                         'L': 72,
                         'N': 73, 'M': 74, 'O': 75, 'P': 76, 'Q': 77, 'R': 78, 'S': 79, 'T': 80, 'U': 81, 'V': 82,
                         'W': 83,
                         'X': 84, 'Y': 85, 'Z': 86, '(가)': 87, '(나)': 88, '(다)': 89, '(라)': 90, '월요일': 91, '화요일': 92,
                         '수요일': 93, '목요일': 94, '금요일': 95, '토요일': 96, '일요일': 97, '농구공': 98, '배구공': 99,
                         '테니스공': 100, '탁구공': 101, '야구공': 102, '축구공': 103, '노란색': 104, '파란색': 105, '빨간색': 106,
                         '보라색': 107, '강아지': 108, '개구리': 109, '거위': 110, '고라니': 111, '고래': 112, '고양이': 113, '곰': 114,
                         '기린': 115, '늑대': 116, '달팽이': 117, '물고기': 118, '병아리': 119, '비둘기': 120, '사자': 121, '여우': 122,
                         '오리': 123, '원숭이': 124, '코끼리': 125, '토끼': 126, '펭귄': 127, '볼펜': 128, '도서관': 129, '박물관': 130,
                         '사탕': 131, '과자': 132, '바나나': 133, '오토바이': 134, '트럭': 135, '자동차': 136, '자전거': 137, '비행기': 138,
                         '버스': 139, '배': 140, '기차': 141, 'number11': 142, 'number12': 143, 'number13': 144,
                         'number14': 145, 'number15': 146,
                         'number16': 147, 'number17': 148, 'number18': 149, 'number19': 150, 'number20': 151,
                         'number21': 152, 'number22': 153,
                         'number23': 154, 'number24': 155, 'number25': 156, 'number26': 157, 'number27': 158,
                         'number28': 159, 'number29': 160}

            self.id2w = {0: '<s>', 1: '</s>', 2: '+', 3: '-', 4: '*', 5: '/', 6: 'number0', 7: 'number1', 8: 'number2',
                         9: 'number3', 10: 'number4', 11: 'number5', 12: 'number6', 13: 'number7',
                         14: 'number8', 15: 'number9', 16: 'number10', 17: '(', 18: ')', 19: 'unk', 20: 'math.comb',
                         21: 'math.factorial', 22: 'max', 23: 'min', 24: '[', 25: ']', 26: '//', 27: '%', 28: '<',
                         29: '>', 30: '>=', 31: '<=',
                         32: '==', 33: '!=', 34: '**', 35: 'if', 36: 'else', 37: 'elif', 38: 'sum', 39: 'bool',
                         40: 'not', 41: 'and', 42: 'or',
                         43: 'abs', 44: 'arr', 45: 'printx for x in range', 46: 'for i in', 47: 'for x in range',
                         48: 'for n in range',
                         49: 'for index in range', 50: '민영', 51: '유나', 52: '정국', 53: '유정', 54: '태형', 55: '남준', 56: '윤기',
                         57: '호석', 58: '지민', 59: '석진', 60: '은지', 61: 'A', 62: 'B', 63: 'C', 64: 'D', 65: 'E', 66: 'F',
                         67: 'G',
                         68: 'H', 69: 'I', 70: 'J', 71: 'K', 72: 'L', 73: 'N', 74: 'M', 75: 'O', 76: 'P', 77: 'Q',
                         78: 'R', 79: 'S',
                         80: 'T', 81: 'U', 82: 'V', 83: 'W', 84: 'X', 85: 'Y', 86: 'Z', 87: '(가)', 88: '(나)', 89: '(다)',
                         90: '(라)', 91: '월요일',
                         92: '화요일', 93: '수요일', 94: '목요일', 95: '금요일', 96: '토요일', 97: '일요일', 98: '농구공', 99: '배구공',
                         100: '테니스공', 101: '탁구공', 102: '야구공', 103: '축구공', 104: '노란색', 105: '파란색', 106: '빨간색',
                         107: '보라색', 108: '강아지', 109: '개구리', 110: '거위', 111: '고라니', 112: '고래', 113: '고양이', 114: '곰',
                         115: '기린', 116: '늑대', 117: '달팽이', 118: '물고기', 119: '병아리', 120: '비둘기', 121: '사자', 122: '여우',
                         123: '오리', 124: '원숭이', 125: '코끼리', 126: '토끼', 127: '펭귄', 128: '볼펜', 129: '도서관', 130: '박물관',
                         131: '사탕', 132: '과자', 133: '바나나', 134: '오토바이', 135: '트럭', 136: '자동차', 137: '자전거',
                         138: '비행기', 139: '버스', 140: '배', 141: '기차', 142: 'number11', 143: 'number12', 144: 'number13',
                         145: 'number14', 146: 'number15', 147: 'number16', 148: 'number17', 149: 'number18',
                         150: 'number19',
                         151: 'number20', 152: 'number21', 153: 'number22', 154: 'number23', 155: 'number24',
                         156: 'number25',
                         157: 'number26', 158: 'number27', 159: 'number28', 160: 'number29'}

            self.w2c = {'+': 0, '-': 0, '*': 0, '/': 0, 'number0': 0, 'number1': 0, 'number2': 0, 'number3': 0,
                        'number4': 0, 'number5': 0, 'number6': 0, 'number7': 0, 'number8': 0, 'number9': 0
                , 'number10': 0, '(': 0, ')': 0, '[': 0, ']': 0, 'unk': 0, 'math.comb': 0, 'math.factorial': 0,
                        'max': 0, 'min': 0, '//': 0, '%': 0, '<': 0, '>': 0, '>=': 0, '<=': 0, '==': 0, '!=': 0,
                        '**': 0,
                        'if': 0, 'else': 0, 'elif': 0, 'sum': 0, 'bool': 0, 'not': 0, 'and': 0, 'or': 0, 'abs': 0,
                        'arr': 0,
                        'printx for x in range': 0, 'for i in': 0, 'for x in range': 0, 'for n in range': 0,
                        'for index in range': 0,
                        '민영': 0, '유나': 0, '정국': 0, '유정': 0, '태형': 0, '남준': 0, '윤기': 0, '호석': 0, '지민': 0, '석진': 0,
                        '은지': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0,
                        'L': 0,
                        'N': 0, 'M': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0,
                        'Y': 0,
                        'Z': 0, '(가)': 0, '(나)': 0, '(다)': 0, '(라)': 0, '월요일': 0, '화요일': 0, '수요일': 0, '목요일': 0,
                        '금요일': 0,
                        '토요일': 0, '일요일': 0, '농구공': 0, '배구공': 0, '테니스공': 0, '탁구공': 0, '야구공': 0, '축구공': 0, '노란색': 0,
                        '파란색': 0, '빨간색': 0, '보라색': 0, '강아지': 0, '개구리': 0, '거위': 0, '고라니': 0, '고래': 0, '고양이': 0,
                        '곰': 0, '기린': 0, '늑대': 0, '달팽이': 0, '물고기': 0, '병아리': 0, '비둘기': 0, '사자': 0, '여우': 0,
                        '오리': 0, '원숭이': 0, '코끼리': 0, '토끼': 0, '펭귄': 0, '볼펜': 0, '도서관': 0, '박물관': 0, '사탕': 0,
                        '과자': 0, '바나나': 0, '오토바이': 0, '트럭': 0, '자동차': 0, '자전거': 0, '비행기': 0, '버스': 0, '배': 0,
                        '기차': 0, 'number11': 0, 'number12': 0, 'number13': 0, 'number14': 0, 'number15': 0,
                        'number16': 0, 'number17': 0,
                        'number18': 0, 'number19': 0, 'number20': 0, 'number21': 0, 'number22': 0, 'number23': 0,
                        'number24': 0,
                        'number25': 0, 'number26': 0, 'number27': 0, 'number28': 0, 'number29': 0}

            self.nwords = 161

    def add_word(self, word):
        if word not in self.w2id:  # IT SHOULD NEVER GO HERE!!
            self.w2id[word] = self.nwords
            self.id2w[self.nwords] = word
            self.w2c[word] = 1
            self.nwords += 1
        else:
            self.w2c[word] += 1

    def add_sent(self, sent):
        for word in sent.split():
            self.add_word(word)
        # elf.add_word(sent)

    def get_id(self, idx):
        return self.w2id[idx]

    def get_word(self, idx):
        return self.id2w[idx]

    def create_vocab_dict(self, args, train_dataloader):
        for data in train_dataloader:
            for sent in data['eqn']:
                self.add_sent(sent)

        assert len(self.w2id) == self.nwords
        assert len(self.id2w) == self.nwords

    def add_to_vocab_dict(self, args, dataloader):
        for data in dataloader:
            for sent in data['eqn']:
                self.add_sent(sent)

        assert len(self.w2id) == self.nwords
        assert len(self.id2w) == self.nwords


def bleu_scorer(ref, hyp, script='default'):
    '''
        Bleu Scorer (Send list of list of references, and a list of hypothesis)
    '''

    refsend = []
    for i in range(len(ref)):
        refsi = []
        for j in range(len(ref[i])):
            refsi.append(ref[i][j].split())
        refsend.append(refsi)

    gensend = []
    for i in range(len(hyp)):
        gensend.append(hyp[i].split())

    if script == 'nltk':
        metrics = corpus_bleu(refsend, gensend)
        return [metrics]

    metrics = compute_bleu(refsend, gensend)
    return metrics