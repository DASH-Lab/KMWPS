from prerequisite import *
from preprocess import *


class TextDataset(Dataset):
    def __init__(self, test, tp, data_path='./data/', datatype='train', max_length=30, is_train=False):
        if datatype == 'train':
            file_df = tp.copy()
            # file_df = file_df[file_df['fold']!=1].reset_index(drop=True)

        else:
            # file_df= tp.copy()
            # file_df = file_df[file_df['fold']==1].reset_index(drop=True)

            file_df = test.copy()
            # file_df = pd.read_csv('./test.csv'')

        self.ques = file_df['Question'].values  # np ndarray of size (#examples,)
        self.eqn = file_df['Equation'].values
        self.nums = file_df['Numbers'].values
        self.names = file_df['Names'].values
        self.ans = file_df['Answer'].values

        self.max_length = max_length

        all_sents = zip(self.ques, self.eqn, self.nums, self.ans)

        if is_train:
            all_sents = sorted(all_sents, key=lambda x: len(x[0].split()))

        self.ques, self.eqn, self.nums, self.ans = zip(*all_sents)

        self.okt = Okt()

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        #         ques = self.process_string(str(self.ques[idx]))
        #         eqn = self.process_string(str(self.eqn[idx]))
        ques = str(self.ques[idx])
        eqn = str(self.eqn[idx])
        nums = self.nums[idx]
        ans = self.ans[idx]
        names = self.names[idx]
        #         print('eqn:', eqn)

        return {'ques': self.curb_to_length(ques), 'eqn': self.curb_to_length(eqn), 'nums': nums, 'ans': ans,
                'names': names}

    def curb_to_length(self, string):
        # todo : set tokenizer(kobert or konlpy)
        return ' '.join(string.strip().split()[:self.max_length])

    def process_string(self, string):
        #         string = re.sub(r"\'s", " 's", string)
        #         string = re.sub(r"\'ve", " 've", string)
        #         string = re.sub(r"n\'t", " n't", string)
        #         string = re.sub(r"\'re", " 're", string)
        #         string = re.sub(r"\'d", " 'd", string)
        #         string = re.sub(r"\'ll", " 'll", string)

        return string


def load_data(config):
    '''
        Loads the data from the datapath in torch dataset form

        Args:
            config (dict) : configuration/args
            logger (logger) : logger object for logging

        Returns:
            dataloader(s)
    '''
    data_path = "./"
    print("Getting test.csv & tp...")
    test, tp = start()

    if config.mode == 'train':
        print('Loading Training Data...')

        '''Load Datasets'''
        train_set = TextDataset(test, tp, data_path=data_path,
                                datatype='train', max_length=config.max_length, is_train=True)
        val_set = TextDataset(test, tp, data_path=data_path,
                              datatype='dev', max_length=config.max_length)

        '''In case of sort by length, write a different case with shuffle=False '''
        train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

        train_size = len(train_dataloader) * config.batch_size
        val_size = len(val_dataloader) * config.batch_size

        msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
        print(msg)

        return train_dataloader, val_dataloader

    elif config.mode == 'test':
        print('Loading Test Data...')

        test_set = TextDataset(data_path=data_path, dataset=config.dataset,
                               datatype='test', max_length=config.max_length)
        test_dataloader = DataLoader(
            test_set, batch_size=config.batch_size, shuffle=False, num_workers=5)

        print('Test Data Loaded...')
        return test_dataloader
