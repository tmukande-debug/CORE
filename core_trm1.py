import torch
from torch import nn
import torch.nn.functional as F
#from recbole.model.layers import TransformerEncoder
import torch
from block_recurrent_transformer_pytorch import*

from core_ave import COREave


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
           enhanced_recurrence = True      # enhanced recurrence from ernie-doc paper, i have seen it to work well on my local machine
          )
        out, mems1, states1 =  self.trm_encoder(item_seq)
        out, mems2, states2 =  self.trm_encoder(item_seq, xl_memories = mems1, states = states1)
        out, mems3, states3 =  self.trm_encoder(item_seq, xl_memories = mems2, states = states2)
        
        num_state_vectors = default(num_state_vectors, block_width)
        xl_memories_layers = default(xl_memories_layers, tuple(range(1, depth + 1)))
        self.xl_memories_layers = set(xl_memories_layers)

        assert all([0 < layer <= depth for layer in xl_memories_layers])

        recurrent_layers = default(recurrent_layers, (depth // 2,)) # default to one recurent layer at middle of the network
        self.recurrent_layers = set(recurrent_layers)

        assert all([0 < layer <= depth for layer in recurrent_layers])

        self.token_emb = nn.Embedding(num_tokens, dim)

        pos_mlp_dim = default(dynamic_pos_bias_dim, dim // 4)
        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(1, pos_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_mlp_dim, pos_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_mlp_dim, heads)
        )

        self.layers = nn.ModuleList([])

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers
            is_xl_layer = layer in self.xl_memories_layers

            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            # only layers with xl memories
            # or has recurrence in horizontal direction
            # use qk rmsnorm (in paper, they use cosine sim attention, but i think qk rmsnorm is more proven given Vit 22B paper)
            # one can also override to use all qk rmsnorm by setting all_layers_qk_rmsnorm = True

            qk_rmsnorm = all_layers_qk_rmsnorm or is_recurrent_layer or is_xl_layer

            self.layers.append(nn.ModuleList([
                AttentionBlock(
                    dim,
                    causal = True,
                    block_width = block_width,
                    dim_head = dim_head,
                    heads = heads,
                    single_head_kv = single_head_kv,
                    qk_rmsnorm = qk_rmsnorm,
                    num_state_vectors = layer_num_state_vectors
                ),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.max_seq_len = max_seq_len
        self.block_width = block_width

        assert divisible_by(max_seq_len, block_width)

        self.ignore_index = ignore_index

        self.enhanced_recurrence = enhanced_recurrence

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
        return_memories_and_states = None  # can force to either return memory + state or not. by default will only return when number of tokens == max_seq_len
    ):
        device = x.device

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # get sequence length i and j for dynamic pos bias

        assert x.shape[-1] <= self.max_seq_len

        w = self.block_width

        # token embedding

        x = self.token_emb(x)

        # dynamic pos bias

        rel_dist = torch.arange(w, dtype = x.dtype, device = device)
        rel_dist = rearrange(rel_dist, '... -> ... 1')
        pos_bias = self.dynamic_pos_bias_mlp(rel_dist)

        i_arange = torch.arange(w, device = device)
        j_arange = torch.arange(w * 2, device = device)
        rel_pos = ((rearrange(i_arange, 'i -> i 1') + w) - rearrange(j_arange, 'j -> 1 j')).abs()

        attn_mask = rel_pos < w  # make sure each token only looks back a block width
        rel_pos = rel_pos.masked_fill(~attn_mask, 0)

        pos_bias = pos_bias[rel_pos]
        pos_bias = rearrange(pos_bias, 'i j h -> h i j')

        # enhanced recurrence

        if self.enhanced_recurrence and len(xl_memories) > 1:
            xl_memories = [*xl_memories[1:], xl_memories[0]]

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

        logits = self.to_logits(x)

        if not return_loss:
            return logits, next_xl_memories, next_states

        logits = rearrange(logits, 'b n c -> b c n')
        loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

        return loss, next_xl_memories, next_states

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_emb):
        mask = item_seq.gt(0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, **kwargs)
        output = trm_output[-1]

        alpha = self.fn(output).to(torch.double)
        alpha = torch.where(mask.unsqueeze(-1), alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        return alpha

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
