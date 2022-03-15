from prerequisite import *


# model
class BertEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', device='cuda', freeze_bert=False):
        super(BertEncoder, self).__init__()
        self.bert_layer = BertForPreTraining(BertConfig(num_hidden_layers=4, hidden_size=252, intermediate_size=786))#.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # self.bert_layer = BertForPreTraining(BertConfig(num_hidden_layers=12))#.from_pretrained(bert_model)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # self.bert_layer = DistilBertModel.from_pretrained(bert_model)
        # self.bert_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        
        # self.bert_layer = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")
        # self.bert_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-discriminator')

        # self.bert_layer = AutoModel.from_pretrained(bert_model)
        # self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)

        # self.bert_layer = BertModel.from_pretrained(bert_model, return_dict=False)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.device = device

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

        # Feed through bert
        # print(self.bert_layer(token_ids, attention_mask=attn_masks).keys())
        # cont_reps, _ = self.bert_layer(token_ids, attention_mask=attn_masks).values()
        cont_reps = self.bert_layer(token_ids, attention_mask=attn_masks)[0]
        # cont_reps = cont_reps['last_hidden_state']

        return cont_reps, token_ids


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
    def __init__(self, config, voc1, voc2, device, EOS_tag='</s>', SOS_tag='<s>'):
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
            self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
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
            src, src_tokens = self.embedding1(ques)
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

        return output

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
                src, _ = self.embedding1(ques)
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