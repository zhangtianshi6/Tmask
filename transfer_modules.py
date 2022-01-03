import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

d_ff = 2048
d_k = d_v = 32

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embedding):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Linear(embedding, d_ff)
        self.conv2 = nn.Linear(d_ff, embedding)
        self.relu1 = nn.ReLU()
        self.norm = nn.LayerNorm(embedding)

    def forward(self, inputs):
        residual = inputs
        output = self.conv1(inputs)
        output = self.relu1(output)
        output = self.conv2(output)
        return self.norm(output + residual)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(embedding, d_k * n_heads)
        self.W_K = nn.Linear(embedding, d_k * n_heads)
        self.W_V = nn.Linear(embedding, d_k * n_heads)
        self.linear1 = nn.Linear(d_k * n_heads, embedding)
        self.norm = nn.LayerNorm(embedding)
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q)
        q_s = q_s.view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, d_v).transpose(1, 2)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.view(batch_size, -1, self.n_heads * d_v)
        output = self.linear1(context)
        return self.norm(output + residual), attn


class EncoderLayer(nn.Module):
    def __init__(self, embedding, heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(embedding, heads)
        self.pos_ffn = PoswiseFeedForwardNet(embedding)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding, heads) for _ in range(nlayers)])

    def forward(self, node_attr):
        for layer in self.layers:
            node_attr, enc_self_attn = layer(node_attr)
        return node_attr


class DecoderLayer(nn.Module):
    def __init__(self, embedding, heads):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(embedding, heads)
        self.pos_ffn = PoswiseFeedForwardNet(embedding)

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_inputs, enc_outputs, enc_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Transformersub(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformersub, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]
        node_num = enc_inputs.shape[2]
        em_len = enc_inputs.shape[3]
        output = torch.zeros_like(enc_inputs)
        en_input = enc_inputs[:, :2].clone()
        de_input = dec_inputs[:, 1:].clone() 
        cat_inputs = torch.cat((en_input, de_input), 1)
        for k in range(1, seq_len+1):
            if k == 1:
                last_inputs = cat_inputs[:, 0]
            enc_input = last_inputs
            dec_input = self.encoder(cat_inputs[:, k])
            for i in range(self.nlayers):
                enc_input = self.decoder(enc_input, dec_input)
            bn = enc_input.size(0)
            seq = enc_input.size(1)
            dec_output = enc_input.view(bn*seq, -1)
            dec_output = self.linear(dec_output).view(bn, seq, -1)
            out_put = dec_output-last_inputs
            last_inputs = dec_output
            out_put = out_put.view(bn*seq, -1)
            out_put = self.linear1(out_put).view(bn, seq, -1)
            output[:, k-1] = out_put
        output = output.view(batch_size*seq_len, node_num, em_len)
        return output

class Transformersubsat(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformersubsat, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]
        node_num = enc_inputs.shape[2]
        em_len = enc_inputs.shape[3]
        output = torch.zeros_like(enc_inputs)
        en_input = enc_inputs[:, :2].clone()
        de_input = dec_inputs[:, 1:].clone() 
        cat_inputs = torch.cat((en_input, de_input), 1)
        for k in range(1, seq_len+1):
            if k == 1:
                last_inputs = self.encoder(cat_inputs[:, 0])
            enc_input = last_inputs
            dec_input = self.encoder(cat_inputs[:, k])
            for i in range(self.nlayers):
                enc_input = self.decoder(enc_input, dec_input)
            bn = enc_input.size(0)
            seq = enc_input.size(1)
            dec_output = enc_input.view(bn*seq, -1)
            dec_output = self.linear(dec_output).view(bn, seq, -1)
            out_put = dec_output-last_inputs
            last_inputs = dec_output
            out_put = out_put.view(bn*seq, -1)
            out_put = self.linear1(out_put).view(bn, seq, -1)
            output[:, k-1] = out_put
        output = output.view(batch_size*seq_len, node_num, em_len)
        return output

class Transformersubsatall(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformersubsatall, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]
        node_num = enc_inputs.shape[2]
        em_len = enc_inputs.shape[3]
        output = torch.zeros_like(enc_inputs)
        en_input = enc_inputs[:, :2].clone()
        de_input = dec_inputs[:, 1:].clone() 
        cat_inputs = torch.cat((en_input, de_input), 1)
        for k in range(1, seq_len+1):
            #if k == 1:
            last_inputs = self.encoder(cat_inputs[:, 0])
            enc_input = last_inputs
            dec_input = self.encoder(cat_inputs[:, k])
            for i in range(self.nlayers):
                enc_input = self.decoder(enc_input, dec_input)
            bn = enc_input.size(0)
            seq = enc_input.size(1)
            dec_output = enc_input.view(bn*seq, -1)
            dec_output = self.linear(dec_output).view(bn, seq, -1)
            out_put = dec_output-last_inputs
            last_inputs = dec_output
            out_put = out_put.view(bn*seq, -1)
            out_put = self.linear1(out_put).view(bn, seq, -1)
            output[:, k-1] = out_put
        output = output.view(batch_size*seq_len, node_num, em_len)
        return output

class Transformersubunsat(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformersubunsat, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]
        node_num = enc_inputs.shape[2]
        em_len = enc_inputs.shape[3]
        output = torch.zeros_like(enc_inputs)
        en_input = enc_inputs[:, :2].clone()
        de_input = dec_inputs[:, 1:].clone() 
        cat_inputs = torch.cat((en_input, de_input), 1)
        for k in range(1, seq_len+1):
            if k == 1:
                last_inputs = cat_inputs[:, 0]#self.encoder(cat_inputs[:, 0])
            enc_input = last_inputs
            dec_input = cat_inputs[:, k]#self.encoder(cat_inputs[:, k])
            for i in range(self.nlayers):
                enc_input = self.decoder(enc_input, dec_input)
            bn = enc_input.size(0)
            seq = enc_input.size(1)
            dec_output = enc_input.view(bn*seq, -1)
            dec_output = self.linear(dec_output).view(bn, seq, -1)
            out_put = dec_output-last_inputs
            last_inputs = dec_output
            out_put = out_put.view(bn*seq, -1)
            out_put = self.linear1(out_put).view(bn, seq, -1)
            output[:, k-1] = out_put
        output = output.view(batch_size*seq_len, node_num, em_len)
        return output


class Transformeradd(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformeradd, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]
        node_num = enc_inputs.shape[2]
        em_len = enc_inputs.shape[3]
        output = torch.zeros_like(enc_inputs)
        en_input = enc_inputs[:, :2].clone()
        de_input = dec_inputs[:, 1:].clone() 
        cat_inputs = torch.cat((en_input, de_input), 1)
        for k in range(1, seq_len+1):
            if k == 1:
                last_inputs = cat_inputs[:, 0]
            enc_input = last_inputs
            dec_input = self.encoder(cat_inputs[:, k])
            for i in range(self.nlayers):
                enc_input = self.decoder(enc_input, dec_input)
            bn = enc_input.size(0)
            seq = enc_input.size(1)
            dec_output = enc_input.view(bn*seq, -1)
            dec_output = self.linear(dec_output).view(bn, seq, -1)
            output[:, k-1] = dec_output
            last_inputs = dec_output+last_inputs
            last_inputs = last_inputs.view(bn*seq, -1)
            last_inputs = self.linear1(last_inputs).view(bn, seq, -1)
        output = output.view(batch_size*seq_len, node_num, em_len)
        return output


class Transformerdiradd(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformerdiradd, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.shape[0]
        seq_len = enc_inputs.shape[1]
        node_num = enc_inputs.shape[2]
        em_len = enc_inputs.shape[3]
        output = torch.zeros_like(enc_inputs)
        en_input = enc_inputs[:, :2].clone()
        de_input = dec_inputs[:, 1:].clone()
        cat_inputs = torch.cat((en_input, de_input), 1)
        for k in range(1, seq_len+1):
            if k == 1:
                last_inputs = cat_inputs[:, 0]
            enc_input = last_inputs
            dec_input = self.encoder(cat_inputs[:, k])
            for i in range(self.nlayers):
                enc_input = self.decoder(enc_input, dec_input)
            bn = enc_input.size(0)
            seq = enc_input.size(1)
            dec_output = enc_input.view(bn*seq, -1)
            dec_output = self.linear(dec_output).view(bn, seq, -1)
            output[:, k-1] = dec_output
            last_inputs = dec_output+last_inputs
        output = output.view(batch_size*seq_len, node_num, em_len)
        return output


class Transformertrans(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformertrans, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.decoder = DecoderLayer(embedding, heads)
        self.linear = nn.Linear(output_dim, output_dim)
        self.nlayers = nlayers

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.size(0)
        seq_num = enc_inputs.size(1)
        num_nodes = enc_inputs.size(2)
        dec_inputs = dec_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        enc_inputs = enc_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        enc_inputs = self.encoder(enc_inputs)
        dec_output = dec_inputs
        for i in range(self.nlayers):
            dec_output = self.decoder(dec_output, enc_inputs)
        dec_output = dec_output.view(batch_size*seq_num*num_nodes, -1)
        dec_output = self.linear(dec_output).view(batch_size*seq_num, num_nodes, -1)
        dec_output = dec_output.reshape(batch_size, seq_num, num_nodes, -1)
        return dec_output


class Transformercat(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformercat, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.linear = nn.Linear(output_dim*2, output_dim)

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.size(0)
        seq_num = enc_inputs.size(1)
        num_nodes = enc_inputs.size(2)
        dec_inputs = dec_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        enc_inputs = enc_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        cat_state = torch.cat((dec_inputs, enc_inputs), -2)
        dec_output = self.encoder(cat_state)
        curr_node = dec_output[:, :num_nodes, :]
        next_node = dec_output[:, num_nodes:, :]
        node_attr = torch.cat((curr_node, next_node), -1)
        node_attr = node_attr.reshape(batch_size*seq_num*num_nodes, -1)
        node_attr = self.linear(node_attr)
        node_attr = node_attr.reshape(batch_size*seq_num, num_nodes, -1)
        node_attr = node_attr.reshape(batch_size, seq_num, num_nodes, -1)

        return node_attr


class Transformercatlstm(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformercatlstm, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.linear = nn.Linear(2*output_dim, output_dim)
        self.lstm = nn.LSTM(output_dim*2, output_dim*2, 2, batch_first=True)

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.size(0)
        seq_num = enc_inputs.size(1)
        num_nodes = enc_inputs.size(2)
        dec_inputs = dec_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        enc_inputs = enc_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        cat_state = torch.cat((dec_inputs, enc_inputs), -2)
        dec_output = self.encoder(cat_state)
        curr_node = dec_output[:, :num_nodes, :]
        next_node = dec_output[:, num_nodes:, :]
        node_attr = torch.cat((curr_node, next_node), -1)

        node_attr = node_attr.view(batch_size, seq_num, num_nodes, -1).transpose(1, 2).reshape(batch_size*num_nodes, seq_num, -1)
        node_attr, _ = self.lstm(node_attr)
        node_attr = node_attr.reshape(batch_size*seq_num*num_nodes, -1)
        node_attr = self.linear(node_attr).reshape(batch_size*num_nodes, seq_num, -1)
        node_attr = node_attr.reshape(batch_size, num_nodes, seq_num, -1).transpose(1, 2)
        return node_attr


class Transformerelstm(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformerelstm, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.linear = nn.Linear(2*output_dim, output_dim)
        self.lstm = nn.LSTM(output_dim, output_dim, 4, batch_first=True)

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.size(0)
        seq_num = enc_inputs.size(1)
        num_nodes = enc_inputs.size(2)
        states = enc_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        next_states = dec_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        states = self.encoder(states)
        next_states = self.encoder(next_states)
        states = states.reshape(batch_size, seq_num, num_nodes, -1).transpose(1, 2).reshape(batch_size*num_nodes, seq_num, -1)
        next_states = next_states.reshape(batch_size, seq_num, num_nodes, -1).transpose(1, 2).reshape(batch_size*num_nodes, seq_num, -1)
        cat_states = torch.cat((states[:, :-1, :], next_states[:, -2:, :]), -2)
        node_attr, _ = self.lstm(cat_states)
        node_attr = node_attr[:, 1:]
        node_attr = node_attr.reshape(batch_size, num_nodes, seq_num, -1).transpose(1, 2)
        return node_attr


class Transformerlstm(nn.Module):
    def __init__(self, embedding, output_dim, heads, nlayers, nodes):
        super(Transformerlstm, self).__init__()
        self.encoder = Encoder(embedding, output_dim, heads, nlayers, nodes)
        self.linear = nn.Linear(2*output_dim, output_dim)
        self.lstm = nn.LSTM(output_dim, output_dim, 4, batch_first=True)

    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.size(0)
        seq_num = enc_inputs.size(1)
        num_nodes = enc_inputs.size(2)
        states = enc_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        next_states = dec_inputs.reshape(batch_size*seq_num, num_nodes, -1)
        states = states.reshape(batch_size, seq_num, num_nodes, -1).transpose(1, 2).reshape(batch_size*num_nodes, seq_num, -1)
        next_states = next_states.reshape(batch_size, seq_num, num_nodes, -1).transpose(1, 2).reshape(batch_size*num_nodes, seq_num, -1)
        cat_states = torch.cat((states[:, :-1, :], next_states[:, -2:, :]), -2)
        node_attr, _ = self.lstm(cat_states)
        node_attr = node_attr[:, 1:]
        node_attr = node_attr.reshape(batch_size, num_nodes, seq_num, -1).transpose(1, 2)
        return node_attr
