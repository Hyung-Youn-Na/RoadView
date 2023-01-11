import torch
import torch.nn.functional as F

from language.GRUFC import GRUFC
from transformers import BartModel, optimization
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from language.base import BaseModel

class Custom_lm(BaseModel):

    def __init__(self):
        # language module
        super().__init__()

        # self.lm_cfg = lm_cfg
        # self.dictmap = mmcv.load(lm_cfg.dictmap_file)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(lm_cfg.bert_vocab_file)
        # self.bert_model = BertModel.from_pretrained(
        #     lm_cfg.bert_model_file, config=BertConfig.from_json_file(lm_cfg.bert_cfg_file))
        self.bart_tokenizer = get_kobart_tokenizer()
        self.bart_model = BartModel.from_pretrained(get_pytorch_kobart_model())
        self.lang_model = GRUFC(input_dim=768, output_dim=3, gru_num=2, with_bi=True)
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')





    def init_weights(self):
        # language module
        # for param in self.bart_model.parameters():
        #     param.requires_grad = False
        self.lang_model.init_weights()

    def forward(self, x):
        tokens = self.get_token(self.bart_tokenizer, x).to(self.device)

        tensor = torch.tensor((), dtype=torch.long)

        segments = tensor.new_ones((tokens.size(0), tokens.size(1)), dtype=torch.long)

        x_nlp = self.bart_model(tokens)

        outputs = self.lang_model(x_nlp.last_hidden_state)

        return outputs[0]


    def loss(self, cls_scores, labels, type=''):
        return self.lang_model.loss(cls_scores, labels, type)


    def get_token(self, bert_tokenizer, texts, max_len=16):
        tokens = []
        for i, text in enumerate(texts):
            text = '[CLS]' + text + '[SEP]'
            tokenized_text = bert_tokenizer.tokenize(text)
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

            token = torch.tensor([indexed_tokens])

            if token.size(1) < max_len:
                pad = max_len - token.size(1)
                token = F.pad(token, (0, pad), 'constant', 0)
            else:
                token = token[:, :max_len]
            tokens.append(token)
        tokens = torch.cat(tokens, 0)

        return tokens

    def forward_train(self, gt_file):
        losses = self.forward_train_nlp(gt_file=gt_file)

        return losses

    def forward_train_nlp(self, gt_file):
        losses = dict()

        x_nlp = []
        labels = []

        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

        with open(gt_file, 'r', encoding='utf-8') as gt:
            lines = gt.readlines()
            for line in lines:
                with torch.no_grad():
                    text, label = line.split('\t')

                    tokens = self.get_token(self.bart_tokenizer, text).to(device)

                    tensor = torch.tensor((), dtype=torch.long)

                    segments = tensor.new_ones((tokens.size(0), tokens.size(1)), dtype=torch.long)
                    x_nlp_i = self.bart_model(tokens, token_type_ids=segments)

                # prepare label
                labels.append(int(label))
                x_nlp.append(x_nlp_i)

        x_nlp = torch.cat(x_nlp, dim=0)
        labels = torch.cat(labels, dim=0)
        outputs = self.lang_model(x_nlp)
        cls_scores = outputs[0]
        loss_lang = self.lang_model.loss(cls_scores, labels, type='lm')
        losses.update(loss_lang)

        return losses