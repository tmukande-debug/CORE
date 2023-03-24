import torch
from torch import nn
import torch.nn.functional as F
#from recbole.model.layers import TransformerEncoder
import torch
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer

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
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fn = nn.Linear(self.hidden_size, 1)

        self.apply(self._init_weights)

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
