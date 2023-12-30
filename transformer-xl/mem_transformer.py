import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    freqs_cis = freqs_cis[-x.shape[0]:]
    shape = [d if i == 0 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_q = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GRMSLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, n_group:int = 1, eps: float = 1e-6):
        super().__init__()
        assert dim % n_group == 0
        self.dim = dim
        self.n_group = n_group 
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        if self.n_group!=1: x = x.view([*x.shape[:-1], self.n_group, -1])
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.n_group!=1: x = x.view([*x.shape[:-2], self.dim])
        return x

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropffn=0., use_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w1 = nn.Linear(d_model, d_inner, bias=use_bias)
        self.w2 = nn.Linear(d_model, d_inner, bias=use_bias)
        self.w3 = nn.Linear(d_inner, d_model, bias=use_bias)

        self.dropffn = nn.Dropout(p=dropffn)

    def forward(self, x):
        return self.w3(self.dropffn(F.silu(self.w1(x)) * self.w2(x)))


class RoPeMultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropatt=0., use_bias=True):
        super(RoPeMultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.q_net = nn.Linear(d_model, n_head * d_head)
        self.k_net = nn.Linear(d_model, n_head * d_head)
        self.v_net = nn.Linear(d_model, n_head * d_head, bias=use_bias)

        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=use_bias)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, x, freqs_cis, attn_mask=None, mem=None):
        qlen, bsz = x.size(0), x.size(1)

        x_q, x_k, x_v = self.q_net(x), self.k_net(x), self.v_net(x)
        new_mem = [x_k, x_v]
        if mem is not None:
            k_mem, v_mem = mem
            x_k = torch.cat([k_mem, x_k], dim=0)
            x_v = torch.cat([v_mem, x_v], dim=0)
        klen = x_k.size(0)

        x_q = x_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        x_k = x_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        x_v = x_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        #### compute attention score
        
        x_q, x_k = apply_rotary_emb(x_q, x_k, freqs_cis=freqs_cis)

        attn_weight = torch.einsum('ibnd,jbnd->ijbn', (x_q, x_k)) * self.scale

        #### compute attention probability
        if attn_mask is not None:
            attn_weight = attn_weight.float().masked_fill(
                attn_mask[:,:,None,None], -float('inf')).type_as(attn_weight)

        # [qlen x klen x bsz x n_head]
        attn_score = self.dropatt(F.softmax(attn_weight, dim=1))

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_score, x_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        return attn_out, new_mem


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_inner, dropout, dropatt=0.0, dropffn=0.0, n_group=1, use_bias=True):
        super(TransformerEncoderLayer, self).__init__()

        self.attn = RoPeMultiHeadAttn(d_model, n_head, d_head, dropatt, use_bias)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropffn, use_bias)

        self.dropout = nn.Dropout(dropout)

        self.attn_layer_norm = GRMSLayerNorm(d_model, n_group=n_group)
        self.ffn_layer_norm = GRMSLayerNorm(d_model, n_group=n_group)

    def forward(self, x, freqs_cis, attn_mask=None, mem=None):

        attn_x = self.attn_layer_norm(x)
        attn_x, new_mem = self.attn(attn_x, freqs_cis,
                                    attn_mask=attn_mask,
                                    mem=mem)
        x = x + self.dropout(attn_x)

        ffn_x = self.ffn_layer_norm(x)
        ffn_x = self.ffn(ffn_x)
        x = x + self.dropout(ffn_x)
        
        return x, new_mem


class MemTransformerLM(nn.Module):

    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dropffn, d_embed=None, n_group=1,
                 attn_span=512, tie_weight=True, use_bias=True):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.d_embed = d_model if d_embed is None else d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.attn_span = attn_span

        self.word_emb = nn.Embedding(n_token, d_embed)

        if self.d_embed != self.d_model:
            self.emb_proj = nn.Linear(d_embed, d_model)
        
        self.freqs_cis = precompute_freqs_cis(d_head, attn_span * 2)

        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model, n_head, d_head, d_inner, dropout,
                    dropatt=dropatt, dropffn=dropffn, n_group=n_group, use_bias=use_bias)
            )
        self.layer_norm = GRMSLayerNorm(d_model, n_group=n_group)

        if self.d_embed != self.d_model:
            self.out_proj = nn.Linear(d_model, d_embed)

        self.out_layer = nn.Linear(d_embed, n_token)
        if tie_weight:
            self.out_layer.weight = self.word_emb.weight

    def _init_mems(self):
        return [None for _ in range(self.n_layer)]

    def _update_mems(self, new_mems, mems):
        assert len(new_mems) == len(mems), 'len(hids) != len(mems)'

        with torch.no_grad():
            merged_mems = []
            for new_mem, mem in zip(new_mems, mems):
                if mem is None:
                    merged_mem = new_mem
                else:
                    merged_mem = [torch.cat([mem[0], new_mem[0]], dim=0),
                               torch.cat([mem[1], new_mem[1]], dim=0)]
                merged_mem[0] = merged_mem[0][-self.attn_span+1:].detach()
                merged_mem[1] = merged_mem[1][-self.attn_span+1:].detach()
                merged_mems.append(merged_mem)

        return merged_mems

    def _forward(self, input_ids, mems=None):
        if mems is None: mems = self._init_mems()

        word_emb = self.word_emb(input_ids)
        if self.d_embed != self.d_model: word_emb = self.emb_proj(word_emb)
        layer_out = torch.dropout(word_emb, p=self.dropout, train=self.training)

        qlen = input_ids.size(0)
        mlen = mems[0][0].size(0) if mems[0] is not None else 0
        klen = mlen + qlen
        # attention mask
        all_ones = layer_out.new_ones(qlen, klen).bool()
        attn_mask = torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, mlen-self.attn_span)
        # position embedding
        freqs_cis = self.freqs_cis[:klen].to(layer_out.device)

        new_mems = []
        for i, layer in enumerate(self.layers):
            layer_out, new_mem = layer(layer_out, freqs_cis, attn_mask=attn_mask, mem=mems[i])
            new_mems.append(new_mem)
        layer_out = self.layer_norm(layer_out)

        new_mems = self._update_mems(new_mems, mems)

        if self.d_embed != self.d_model: layer_out = self.out_proj(layer_out)
        logit = self.out_layer(layer_out)

        return logit, new_mems

    def forward(self, input_ids, target=None, mems=None):

        logit, new_mems = self._forward(input_ids, mems)
        if target is not None:
            loss = -F.log_softmax(logit, dim=-1).gather(-1, target.unsqueeze(-1))
            return logit, new_mems, loss
        return logit, new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=4, help='')
    parser.add_argument('--d_head', type=int, default=64, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--d_embed', type=int, default=256, help='')
    parser.add_argument('--d_inner', type=int, default=512, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    seq_len, attn_span = 36, 36
    data_len = seq_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, seq_len, device=device, ext_len=0)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                        args.d_model, args.d_head, args.d_inner, args.dropout,
                        attn_span=attn_span, tie_weight=True, use_bias=True).to(device)

        for name, param in model.named_parameters():
            print(name, param.size())

        mems = None
        for idx, (inp, tgt, seqlen) in enumerate(diter):
            print('batch {}'.format(idx))
            out = model(inp, tgt, mems)
            mems = out[1]
        