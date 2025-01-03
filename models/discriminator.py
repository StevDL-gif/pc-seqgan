import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
import torch.nn.functional as F
import torch.nn.init as init
# class Discriminator(nn.Module):
#
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
#         super(Discriminator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.embedding_dim = embedding_dim
#         self.max_seq_len = max_seq_len
#         self.gpu = gpu
#
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
#         self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
#         self.dropout_linear = nn.Dropout(p=dropout)
#         self.hidden2out = nn.Linear(hidden_dim, 1)
#
#     def init_hidden(self, batch_size):
#         h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))
#
#         if self.gpu:
#             return h.cuda()
#         else:
#             return h
#
#     def forward(self, input, hidden):
#         # input dim                                                # batch_size x seq_len
#         emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
#         emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
#         _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
#         hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
#         out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
#         out = torch.tanh(out)
#         out = self.dropout_linear(out)
#         out = self.hidden2out(out)                                 # batch_size x 1
#         out = torch.sigmoid(out)
#         return out
#
#     def batchClassify(self, inp):
#         """
#         Classifies a batch of sequences.
#
#         Inputs: inp
#             - inp: batch_size x seq_len
#
#         Returns: out
#             - out: batch_size ([0,1] score)
#         """
#
#         h = self.init_hidden(inp.size()[0])
#         out = self.forward(inp, h)
#         return out.view(-1)
#
#     def batchBCELoss(self, inp, target):
#         """
#         Returns Binary Cross Entropy Loss for discriminator.
#
#          Inputs: inp, target
#             - inp: batch_size x seq_len
#             - target: batch_size (binary 1/0)
#         """
#
#         loss_fn = nn.BCELoss()
#         h = self.init_hidden(inp.size()[0])
#         out = self.forward(inp, h)
#         target = target.view(-1, 1)
#         return loss_fn(out, target)

# class Discriminator(nn.Module):
#     """A CNN for text classification
#
#     architecture: Embedding >> Convolution >> Max-pooling >> Softmax
#     """
#
#     def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
#         super(Discriminator, self).__init__()
#         self.emb = nn.Embedding(vocab_size, emb_dim)
#         self.convs = nn.ModuleList([
#             nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
#         ])
#         self.highway = nn.Linear(sum(num_filters), sum(num_filters))
#         self.dropout = nn.Dropout(p=dropout)
#         self.lin = nn.Linear(sum(num_filters), num_classes)
#         self.softmax = nn.LogSoftmax(dim=1)
#         self.init_parameters()
#
#     def forward(self, x):
#         """
#         Args:
#             x: (batch_size * seq_len)
#         """
#         emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
#         convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
#         pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
#         pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
#         highway = self.highway(pred)
#         pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
#         pred = self.softmax(self.lin(self.dropout(pred)))
#         return pred
#
#     def init_parameters(self):
#         for param in self.parameters():
#             param.data.uniform_(-0.05, 0.05)

class Discriminator(nn.Module):
    """
    A Discriminator for sequence data that combines CNN and RNN for better feature extraction.
    Architecture: Embedding >> Convolution >> RNN (GRU) >> Fully connected layers >> Softmax
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda=0.0):
        super(Discriminator, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, num_filter, (filter_size, embedding_size))
            for filter_size, num_filter in zip(filter_sizes, num_filters)
        ])

        self.highway = Highway(sum(num_filters), sum(num_filters))

        self.dropout = nn.Dropout(dropout_keep_prob)

        self.criterion = nn.CrossEntropyLoss()
        # Fully connected layer
        self.fc = nn.Linear(sum(num_filters), num_classes)

        self.batch_norm = nn.LayerNorm(sum(num_filters))
        # Softmax for output
        self.active = nn.LeakyReLU(0.02)
        self.softmax = nn.LogSoftmax(dim=1)
        #self.init_parameters()

    def forward(self, x):
        # Embedding lookup
        x_embedded = self.embedding(x).unsqueeze(1)

        # Convolution and Max Pool
        conv_results = [
            F.relu(conv(x_embedded)).squeeze(3) for conv in self.conv_layers
        ]

        pooled_outputs = [F.max_pool1d(result, result.size(2)).squeeze(2) for result in conv_results]
        concatenated = torch.cat(pooled_outputs, 1)

        # Highway layer
        highway = self.highway(concatenated)

        # Dropout
        dropout = self.dropout(highway)

        # Fully connected layer
        logits = self.fc(dropout)
        probabilities = F.softmax(logits, dim=-1)

        return logits


class Highway(nn.Module):
    def __init__(self, size, num_layers, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(1)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(1)])

    def forward(self, x):
        for i in range(1):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x
