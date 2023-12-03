import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.adaptive_embedding import AdaptiveEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits


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
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFeedForward, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_inner), 
            nn.ReLU(inplace=True),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, input):
        if self.pre_lnorm:
            output = self.linear(self.layer_norm(input))
            output = output + input
        else:
            output = self.linear(input)
            output = self.layer_norm(input + output)
        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 attn_span=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.attn_span = attn_span

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w), dtype=torch.bool)
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.v_rpe = nn.Parameter(torch.zeros(self.attn_span, self.n_head, self.d_head))

    def forward(self, w, r, r_w_bias, r_r_bias, rel_pos=None, rel_pos_mask=None, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        if rel_pos is not None and rel_pos_mask is not None:
            rel_attn_prob = torch.gather(attn_prob, dim=1, index=rel_pos)
            attn_vec = attn_vec + torch.einsum('isbn,snd->ibnd', (rel_attn_prob, self.v_rpe))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout, 
                                    pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, rel_pos=None, rel_pos_mask=None, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               rel_pos=rel_pos, rel_pos_mask=rel_pos_mask,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        
        output = self.feed_forward(output)

        return output


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, d_embed=None, pre_lnorm=False, attn_span=512,
                 cutoffs=[], div_val=1, tie_projs=[False], tie_weight=True):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.attn_span = attn_span

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    attn_span=attn_span, dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                cutoffs, div_val=div_val)

        if tie_weight:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and div_val == 1 and d_model != d_embed:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                elif tie_proj and div_val != 1:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[i]

    def _init_mems(self):
        return [None for _ in range(self.n_layer+1)]

    def _update_mems(self, hids, mems):
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        with torch.no_grad():
            new_mems = []
            for hid, mem in zip(hids, mems):
                if mem is None:
                    new_mem = hid
                else:
                    new_mem = torch.cat([mem, hid], dim=0)
                new_mem = new_mem[-self.attn_span+1:].detach()
                new_mems.append(new_mem)

        return new_mems

    def _forward(self, input_ids, mems=None):
        qlen, bsz = input_ids.size()

        word_emb = self.word_emb(input_ids)

        mlen = mems[0].size(0) if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen
        all_ones = word_emb.new_ones(qlen, klen).bool()
        dec_attn_mask = (torch.triu(all_ones, 1+mlen) + 
                         torch.tril(all_ones, mlen-self.attn_span))[:, :, None] # -1

        pos_seq = torch.arange(klen-1, -1, -1, device=word_emb.device,
                               dtype=torch.long).clamp(max=self.attn_span)
        pos_emb = self.pos_emb(pos_seq)

        rel_pos = torch.arange(qlen, device=word_emb.device, dtype=torch.long)[:,None] + \
                    torch.arange(mlen, mlen-self.attn_span, -1, device=word_emb.device, dtype=torch.long)
        rel_pos_mask = (rel_pos >= 0).type_as(pos_emb)
        rel_pos_mask = rel_pos_mask.view(qlen, self.attn_span, 1, 1).expand(-1, -1, bsz, self.n_head)
        rel_pos = rel_pos.clamp(min=0)
        rel_pos = rel_pos.view(qlen, self.attn_span, 1, 1).expand(-1, -1, bsz, self.n_head)
        

        layer_out = torch.dropout(word_emb, p=self.dropout, train=self.training)
        pos_emb = torch.dropout(pos_emb, p=self.dropout, train=self.training)

        hids = [layer_out]
        for i, layer in enumerate(self.layers):
            layer_out = layer(layer_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, rel_pos=rel_pos, rel_pos_mask=rel_pos_mask,
                    dec_attn_mask=dec_attn_mask, mems=mems[i])
            hids.append(layer_out)

        layer_out = torch.dropout(layer_out, p=self.dropout, train=self.training)

        new_mems = self._update_mems(hids, mems)

        return layer_out, new_mems

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self._init_mems()

        seq_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        loss = self.crit(hidden.contiguous().view(-1, hidden.size(-1)), target.contiguous().view(-1))
        loss = loss.view(seq_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


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
                        dropatt=args.dropout, tie_weight=True, 
                        d_embed=args.d_embed, div_val=div_val, 
                        tie_projs=tie_projs, pre_lnorm=True,
                        attn_span=attn_span, cutoffs=cutoffs).to(device)

        for name, param in model.named_parameters():
            print(name, param.size())

        mems = tuple()
        for idx, (inp, tgt, seqlen) in enumerate(diter):
            print('batch {}'.format(idx))
            out = model(inp, tgt, *mems)
            mems = out[1:]
        