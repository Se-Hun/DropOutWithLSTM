import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class IMDBLstm(nn.Module):

    def __init__(self, vocab, hidden_size, n_cat, bs=1, nl=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.bs = bs
        self.nl = nl
        self.emb = nn.Embedding(vocab, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, nl)
        self.fc2 = nn.Linear(hidden_size, n_cat)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inp):
        bs = inp.size()[1]
        if bs != self.bs:
            self.bs = bs
        e_out = self.emb(inp)
        h0 = c0 = Variable(e_out.data.new(*(self.nl, self.bs, self.hidden_size)).zero_())
        rnn_o, _ = self.rnn(e_out, (h0, c0))
        rnn_o = rnn_o[-1]
        fc = F.dropout(self.fc2(rnn_o), p=0.5)
        print(fc)
        # 여기 부분에 dropout 결과를 저장하는 코드를 넣어서 살펴봐야할듯?
        return self.softmax(fc)