__author__ = 'max'

from overrides import overrides
import math
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

from wolf.nnet.layer_norm import LayerNorm


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention
    """
    def __init__(self, model_dim, heads, dropout=0.0, mask_diag=False):
        """

        Args:
            model_dim: int
                the input dimension for keys, queries and values
            heads: int
                number of heads
            dropout: float
                dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // heads
        self.heads = heads
        self.dropout = dropout
        self.mask_diag = mask_diag
        assert self.head_dim * heads == self.model_dim, "model_dim must be divisible by number of heads"
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight = Parameter(torch.empty(3 * model_dim, model_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * model_dim))
        self.layer_norm = LayerNorm(model_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # in proj
        nn.init.xavier_uniform_(self.in_proj_weight[:self.model_dim, :])
        nn.init.xavier_uniform_(self.in_proj_weight[self.model_dim:(self.model_dim * 2), :])
        nn.init.xavier_uniform_(self.in_proj_weight[(self.model_dim * 2):, :])
        nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, key_mask=None):
        """

        Args:
            query: Tenfor
                [batch, tgt_len, model_dim]
            key: Tensor
                [batch, src_len, model_dim]
            value: Tensor
                [batch, src_len, model_dim]
            key_mask: ByteTensor or None
                binary ByteTensor [batch, src_len] padding elements are indicated by 1s.

        Returns:

        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        bs, src_len, model_dim = key.size()
        tgt_len = query.size(1)
        heads = self.heads
        residual = query

        # k, v: [bs, src_len, model_dim]
        # q: [bs, tgt_len, model_dim]
        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)
        q *= self.scaling

        model_dim = q.size(2)
        dim = model_dim // heads

        # [len, batch, model_dim] -> [len, batch * heads, dim] -> [batch * heads, len, dim]
        q = q.transpose(0, 1).contiguous().view(tgt_len, bs * heads, dim).transpose(0, 1)
        k = k.transpose(0, 1).contiguous().view(src_len, bs * heads, dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(src_len, bs * heads, dim).transpose(0, 1)

        # attention weights [batch * heads, tgt_len, src_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if key_mask is not None:
            attn_weights = attn_weights.view(bs, heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bs * heads, tgt_len, src_len)

        if self.mask_diag:
            assert tgt_len == src_len
            # [1, tgt_len, tgt_len]
            diag_mask = torch.eye(tgt_len, device=query.device, dtype=torch.uint8).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(diag_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights.float(), dim=-1,
                                 dtype=torch.float32 if attn_weights.dtype == torch.float16 else attn_weights.dtype)


        # outputs [batch * heads, tgt_len, dim]
        out = torch.bmm(attn_weights, v)
        # merge heads
        # [batch, heads, tgt_len, dim] -> [batch, tgt_len, heads, dim]
        # -> [batch, tgt_len, model_dim]
        out = out.view(bs, heads, tgt_len, dim).transpose(1, 2).contiguous().view(bs, tgt_len, model_dim)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.layer_norm(out + residual)
        return out

    def init(self, query, key, value, key_mask=None, init_scale=1.0):
        with torch.no_grad():
            return self(query, key, value, key_mask=key_mask)

    def _in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def _in_proj_kv(self, key):
        return self._in_proj(key, start=self.model_dim).chunk(2, dim=-1)

    def _in_proj_q(self, query):
        return self._in_proj(query, end=self.model_dim)

    def _in_proj_k(self, key):
        return self._in_proj(key, start=self.model_dim, end=2 * self.model_dim)

    def _in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.model_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class MultiHeadAttention2d(nn.Module):
    def __init__(self, channels, heads, dropout=0.0):
        super(MultiHeadAttention2d, self).__init__()
        self.proj = nn.Conv2d(channels, 3 * channels, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout, inplace=True)
        else:
            self.dropout = None
        assert channels % heads == 0
        self.features = channels
        self.heads = heads

    @overrides
    def forward(self, x, pos_enc=None):
        # [batch, channels, height, width]
        if pos_enc is not None:
            x = x + pos_enc
        bs, channels, height, width = x.size()
        heads = self.heads
        dim = channels // heads
        # [batch, 3 * channels, height, width]
        c = self.proj(x)
        # [batch, 3, heads, dim, height, width]
        c = c.view(bs, 3, heads, dim, height, width)
        # [batch, heads, dim, height, width]
        queries = c[:, 0]
        keys = c[:, 1]
        # [batch, heads, dim, timesteps]
        values = c[:, 2].view(bs, heads, dim, height * width)
        # attention weights [batch, heads, height, width, height, width]
        attn_weights = torch.einsum('bhdij,bhdkl->bhijkl', (queries, keys)).div(math.sqrt(dim))
        # attention weights [batch, heads, height, width, timesteps]
        attn_weights = self.softmax(attn_weights.view(bs, heads, height, width, -1))
        # values [batch, heads, dim, height, width]
        out = torch.einsum('bhdt,bhijt->bhdij', (values, attn_weights))
        if self.dropout is not None:
            out = self.dropout(out)
        # merge heads
        # [batch, channels, heads, dim]
        out = x + out.view(bs, channels, height, width)
        return out

    def init(self, x, pos_enc=None, init_scale=1.0):
        with torch.no_grad():
            return self(x, pos_enc=pos_enc)
