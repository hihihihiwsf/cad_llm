from .deepcad_layers.transformer import *
from .deepcad_layers.improved_transformer import *
from .deepcad_layers.positional_encoding import *
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')
EXT_IDX = ALL_COMMANDS.index('Ext')

MAX_N_EXT = 10 # maximum number of extrusion
MAX_N_LOOPS = 6 # maximum number of loops per sketch
MAX_N_CURVES = 15 # maximum number of curves per loop
MAX_TOTAL_LEN = 60 # maximum cad sequence length
ARGS_DIM = 256

def _make_seq_first(*args):
    # N, S, ... -> S, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)


def _make_batch_first(*args):
    # S, N, ... -> N, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)


def _get_key_padding_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    with torch.no_grad():
        key_padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) > 0

        if seq_dim == 0:
            return key_padding_mask.transpose(0, 1)
        return key_padding_mask


def _get_padding_mask(commands, seq_dim=0, extended=False):
    with torch.no_grad():
        padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()

        if extended:
            # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S-3).add_(torch.narrow(padding_mask, seq_dim, 0, S-3)).clamp_(max=1)

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask


def _get_group_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    with torch.no_grad():
        # group_mask = (commands == SOS_IDX).cumsum(dim=seq_dim)
        group_mask = (commands == EXT_IDX).cumsum(dim=seq_dim)
        return group_mask


def _get_visibility_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    S = commands.size(seq_dim)
    with torch.no_grad():
        visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) < S - 1

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask


def _get_key_visibility_mask(commands, seq_dim=0):
    S = commands.size(seq_dim)
    with torch.no_grad():
        key_visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) >= S - 1

        if seq_dim == 0:
            return key_visibility_mask.transpose(0, 1)
        return key_visibility_mask


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _sample_categorical(temperature=0.0001, *args_logits):
    if len(args_logits) == 1:
        arg_logits, = args_logits
        return Categorical(logits=arg_logits / temperature).sample()
    return (*(Categorical(logits=arg_logits / temperature).sample() for arg_logits in args_logits),)


def _threshold_sample(arg_logits, threshold=0.5, temperature=1.0):
    scores = F.softmax(arg_logits / temperature, dim=-1)[..., 1]
    return scores > threshold



class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()

        self.command_embed = nn.Embedding(5, 256)

        args_dim = ARGS_DIM + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, 256)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape

        # src = self.command_embed(commands.long()) + \
        #       self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        # if self.use_group:
        #     src = src + self.group_embed(groups.long())
        src = self.embed_fcn(self.arg_embed((args + 1).long())

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = MAX_TOTAL_LEN
        self.use_group = True
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, mask):
        # padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        # group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        #src = self.embedding(commands, args, group_mask)
        src = self.embedding(commands, commands)

        memory = self.encoder(src, mask=mask, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z


class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)


class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer, self).__init__()

        self.args_dim = ARGS_DIM + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)

    # def forward(self, commands_enc, args_enc,
    #             z=None, return_tgt=True, encode_mode=False):
    def forward(self, input_ids, attention_mask, labels=None, z=None, encode_mode=False):
        commands_enc = input_ids
        args_enc = input_ids
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None

        if z is None:
            z = self.encoder(input_ids, attention_mask)
            z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        if encode_mode: return _make_batch_first(z)

        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1],
            "tgt_args": labels,
            "tgt_commands": None
        }

        return res
