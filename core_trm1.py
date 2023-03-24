import torch
from torch import nn
import torch.nn.functional as F
#from recbole.model.layers import TransformerEncoder
import torch
from block_recurrent_transformer_pytorch import*
from torch import nn, einsum

from einops import rearrange, repeat

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, List, Tuple

from core_ave import COREave


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# bias-less layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# geglu feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


class TransNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.n_layers = config['n_layers']
        self.heads = config['heads']
        self.num_tokens = config['num_tokens']
        self.dim = config['dim']
        self.depth = config['depth']
        self.dim_head = config['dim_head']
        self.max_seq_len= config['max_seq_len']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.block_width = config['block_width']
        self.xl_memories_layers = config['xl_memories_layers']
        self.num_state_vectors = config['num_state_vectors']
        self.recurrent_layers = config['recurrent_layers']
        self.recurrent_layers = config['recurrent_layers']
        self.nhanced_recurrence = config['nhanced_recurrence']
        self.hidden_size = config['embedding_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.output_all_encoded_layers = config['output_all_encoded_layers']
        #num_state_vectors = default(num_state_vectors, block_width)
        #xl_memories_layers = default(xl_memories_layers, tuple(range(1, depth + 1)))
        #self.xl_memories_layers = set(xl_memories_layers)
        #dynamic_pos_bias_dim = self.dynamic_pos_bias_dim
        self.layers = nn.ModuleList([])

        self.position_embedding = nn.Embedding(dataset.field2seqlen['item_id_list'], self.hidden_size)
        self.trm_encoder = BlockRecurrentTransformer(
           num_tokens = 20000,             # vocab size
           dim = 512,                      # model dimensions
           depth = 6,                      # depth
           dim_head = 64,                  # attention head dimensions
           heads = 8,                      # number of attention heads
           max_seq_len = 1024,             # the total receptive field of the transformer, in the paper this was 2 * block size
           block_width = 512,              # block size - total receptive field is max_seq_len, 2 * block size in paper. the block furthest forwards becomes the new cached xl memories, which is a block size of 1 (please open an issue if i am wrong)
           xl_memories_layers = (5, 6),    # which layers to use xl memories. very old deepmind papers have shown you only need the last penultimate layers to have cached key values to see majority of benefit
           num_state_vectors = 512,        # number of state vectors, i believe this was a single block size in the paper, but can be any amount
           recurrent_layers = (4,),        # where to place the recurrent layer(s) for states with fixed simple gating
           enhanced_recurrence = True,
           #dynamic_pos_bias_dim = None     # enhanced recurrence from ernie-doc paper, i have seen it to work well on my local machine
          )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)
        

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        length = None,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        temperature = 1.,
        filter_thres = 0.9,
        return_memories_and_states = False
    ):
        length = default(length, self.max_seq_len + 1)
        start_len = prime.shape[-1]

        assert start_len < self.max_seq_len
        assert length <= (self.max_seq_len + 1)
        assert start_len < length

        output = prime

        memories = []
        states = []

        for ind in range(length - start_len):

            logits, next_memories, next_states = self.forward(
                output,
                xl_memories = xl_memories,
                states = states
            )

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            output = torch.cat((output, sampled), dim = -1)

            if divisible_by(output.shape[-1] - 1, self.max_seq_len): # on the sampling of the last token in the current window, set new memories and states
                memories = next_memories
                states = next_states

        output = output[:, start_len:]

        if return_memories_and_states:
            return output, memories, states

        return output

    def forward(
        self,
        x,
        return_loss = False,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        return_memories_and_states = True 
 # can force to either return memory + state or not. by default will only return when number of tokens == max_seq_len
    ):
        device = x.device

        
        x, labels = x[:, :-1], x[:, 1:]
        
        # get sequence length i and j for dynamic pos bias

        assert x.shape[-1] <= self.max_seq_len

        w = self.block_width

        # token embedding

        #x = self.token_emb(x)

        # dynamic pos bias

        #rel_dist = torch.arange(w, dtype = x.dtype, device = device)
        #rel_dist = rearrange(rel_dist, '... -> ... 1')
        #pos_bias = self.dynamic_pos_bias_mlp(rel_dist)

        # ready xl memories and states

        xl_memories = iter(xl_memories)
        states = iter(states)

        next_xl_memories = []
        next_states = []

        return_memories_and_states = default(return_memories_and_states, self.max_seq_len == x.shape[-2])

        # go through layers

        for ind, (attn, ff) in enumerate(self.layers):

            # determine if the layer requires transformer xl memories

            layer = ind + 1

            is_xl_layer     = layer in self.xl_memories_layers
            is_state_layer  = attn.is_recurrent_layer

            # whether to pass in xl memories

            attn_kwargs = dict(
                rel_pos_bias = pos_bias,
                attn_mask = attn_mask,
                return_memories_and_states = return_memories_and_states
            )

            if is_xl_layer:
                attn_kwargs.update(xl_memories = next(xl_memories, None))

            if is_state_layer:
                attn_kwargs.update(states = next(states, None))

            # attention layer

            residual = x
            attn_branch_out, layer_xl_memories, layer_next_states = attn(x, **attn_kwargs)

            if return_memories_and_states:
                # save states if needed

                if exists(layer_next_states):
                    next_states.append(layer_next_states.detach())

                # save current xl memories if needed

                if is_xl_layer:
                    next_xl_memories.append(layer_xl_memories.detach())

            x = attn_branch_out + residual

            # feedforward layer

            x = ff(x) + x
        #if return_loss:

        loss = torch.all(torch.gt(alpha, self.theta))
        return alpha, loss
        logits = self.to_logits(x)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class COREtrm(COREave):
    def __init__(self, config, dataset):
        super(COREtrm, self).__init__(config, dataset)
        self.net = TransNet(config, dataset)

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.sess_dropout(x)
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output
