import warnings
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from braindecode.models.base import EEGModuleMixin

class BrainStack(EEGModuleMixin, nn.Module):
    """
    This model includes:
      - 7 regional branches (from eeg_areas): CNet
      - 1 global branch (conformer_blocks["all"]): CTNet

    All 8 branch outputs are finally fused via gated expert fusion.

    """
    def __init__(
        self,
        n_outputs=None, 
        n_chans=None,
        eeg_areas={},
        num_areas=8,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        return_features=False,
        activation: nn.Module = nn.ELU,
        activation_transfor: nn.Module = nn.GELU,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.eeg_areas = eeg_areas
        self.n_classes = n_outputs

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if not (self.n_chans <= 64):
            warnings.warn(
                "This model has only been tested on no more than 64 channels.",
                UserWarning,
            )

        self.conformer_blocks = nn.ModuleDict({
            area: nn.Sequential(
                CNet(
                    chans=len(channels), 
                    classes=24,
                    time_points=self.n_times)
            ) for area, channels in self.eeg_areas.items()
        })


        self.all_channels = sorted(list(set(sum(eeg_areas.values(), []))))
        if final_fc_length == "auto":
            final_fc_length_all = self.get_fc_size(len(self.all_channels))
        else:
            final_fc_length_all = final_fc_length

        self.conformer_blocks["all"] = nn.Sequential(
            _PatchEmbedding(
                n_filters_time=n_filters_time,
                filter_time_length=filter_time_length,
                n_channels=self.n_chans,
                pool_time_length=pool_time_length,
                stride_avg_pool=pool_time_stride,
                drop_prob=drop_prob,
                activation=activation,
            ),
            _TransformerEncoder(
                att_depth=att_depth,
                emb_size=n_filters_time,
                att_heads=att_heads,
                att_drop=att_drop_prob,
                activation=activation_transfor,
            ),
            _FullyConnected(final_fc_length=final_fc_length_all, activation=activation),
            _FinalLayer(n_classes=self.n_outputs, return_features=return_features),
        )

        # Soft Weighting Logits Fusion
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.eeg_areas) + 1))
        # self.feature_projector = None
        self.feature_projector = nn.Linear(32, 224)

        # Feature Level Fusion
        self.meta_learner = nn.Sequential(
            nn.Linear(224, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_classes)
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, num_channels, time_steps)
        """
        trans_token_list = []
        regional_features = []
        regional_out = []

        for area, channels in self.eeg_areas.items():
            area_data = x[:, channels, :].unsqueeze(1)  # (B, 1, num_area_channels, time_steps)
            out, feature, trans_token = self.conformer_blocks[area](area_data)   
            regional_features.append(feature)
            trans_token_list.append(trans_token)
            regional_out.append(out)

        all_data = x[:, self.all_channels, :].unsqueeze(1)
        out_global_conformer, feature_all, trans_token_all = self.conformer_blocks["all"](all_data)
        trans_token_list.append(trans_token_all)

        # Project global feature
        projected_global = self.feature_projector(feature_all)

        all_features = regional_features + [projected_global]
        stacked_features = torch.stack(all_features, dim=0)

        # Initial Fusion Strategy (alternative implementation)
        # weights = F.softmax(self.ensemble_weights, dim=0)
        # branch_contributions = {
        #     area: round(weights[idx].item(), 2)
        #     for idx, area in enumerate(list(self.eeg_areas.keys()) + ["global"])
        # }      
        # fused_features = torch.einsum("a,abc->bc", weights, stacked_features)  # (B, C)

        # Improved Fusion Strategy with dropout and temperature
        T = getattr(self, "fusion_temperature", 1.5) 
        dropout_ratio = getattr(self, "fusion_dropout_ratio", 0.25)  # e.g. dropout 25%

        num_branches = stacked_features.shape[0]

        if self.training and dropout_ratio > 0:
            # Randomly keep a subset of branches
            num_keep = max(1, int(num_branches * (1 - dropout_ratio)))
            keep_idx = torch.randperm(num_branches)[:num_keep]
            mask = torch.zeros_like(self.ensemble_weights)
            mask[keep_idx] = 1.0
            masked_weights = self.ensemble_weights.masked_fill(mask == 0, -1e9)
        else:
            masked_weights = self.ensemble_weights

        weights = F.softmax(masked_weights / T, dim=0)  # (num_branches,)

        # Visualize branch contributions (detach and round)
        branch_contributions = {
            area: round(weights[idx].item(), 2)
            for idx, area in enumerate(list(self.eeg_areas.keys()) + ["global"])
        }

        fused_features = torch.einsum("a,abc->bc", weights, stacked_features)  # (B, C)

        # MLP for logits output
        fused_logits = self.meta_learner(fused_features)  # shape: [B, C]
 
        return {
            "distill_teacher": out_global_conformer,  # Global branch output
            "area_logits": regional_out,              # Regional branch outputs
            "all_logits": out_global_conformer,       # Global Conformer output
            "out": fused_logits,                      # Token-based meta-learner fused output
            "branch_weights": branch_contributions,   # Branch contribution weights (dict)
            "branch_weights_tensor": weights.detach() # Branch weights tensor for further loss processing
        }


    def get_fc_size(self, n_chans=None):
        patch_embedding = _PatchEmbedding(
            n_filters_time=40,
            filter_time_length=25,
            n_channels=n_chans,
            pool_time_length=75,
            stride_avg_pool=15,
            drop_prob=0.5,
            activation=nn.ELU,
        )
        out = patch_embedding(torch.ones((1, 1, n_chans, self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]
        return size_embedding_1 * size_embedding_2


class CNet(nn.Module): 
    def __init__(self, chans=22, classes=4, time_points=1001, temp_kernel=32,
                 f1=16, f2=32, d=2, pk1=8, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.25):
        super(CNet, self).__init__()
        # Calculating FC input features
        linear_size = (time_points//(pk1*pk2))*f2

        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False), # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16),  groups=f2, bias=False, padding='same'), # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False), # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        b, c, _, t = x.shape

        if x.size(2) == 1:
            x = x.squeeze(2)  
        else:
            raise ValueError(f"Unexpected shape for x: {x.shape}, expected size of dimension 2 to be 1.")

        token = x.permute(0, 2, 1)  # → (B, T', C)
        features = self.flatten(x)
        logits = self.fc(features)
        return logits, features, token


class _PatchEmbedding(nn.Module):
    """Patch Embedding.
    """

    def __init__(
        self,
        n_filters_time,
        filter_time_length,
        n_channels,
        pool_time_length,
        stride_avg_pool,
        drop_prob,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)
            ),
            # pooling acts as slicing to obtain 'patch' along the
            # time dimension as in ViT
            nn.Dropout(p=drop_prob),
            nn.Identity()  
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),

        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p, activation: nn.Module = nn.GELU):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        att_heads,
        att_drop,
        forward_expansion=4,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=att_drop,
                        activation=activation,
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.
    Similar to the layers used in ViT.
    """

    def __init__(
        self, att_depth, emb_size, att_heads, att_drop, activation: nn.Module = nn.GELU
    ):
        super().__init__(
            *[
                _TransformerEncoderBlock(
                    emb_size, att_heads, att_drop, activation=activation
                )
                for _ in range(att_depth)
            ]
        )


class _FullyConnected(nn.Module):
    def __init__(
        self,
        final_fc_length,
        drop_prob_1=0.5,
        drop_prob_2=0.3,
        out_channels=256,
        hidden_channels=32,
        activation: nn.Module = nn.ELU,
    ):
        """Fully-connected layer for the transformer encoder."""

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            activation(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            activation(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x_reshape = x.contiguous().view(x.size(0), -1) #shape: (batch_size, feature)
        out = self.fc(x_reshape)
        return (out, x)


class _FinalLayer(nn.Module):
    def __init__(
        self,
        n_classes,
        hidden_channels=32,
        return_features=False,
    ):
        """Classification head for the transformer encoder.

        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        """
        super().__init__()
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_channels, n_classes),
        )
        self.return_features = return_features
        classification = nn.Identity()
        if not self.return_features:
            self.final_layer.add_module("classification", classification)

    def forward(self, x):
        input, trans_token = x
        if self.return_features:
            out = self.final_layer(input)
            return out, input, trans_token
        else:
            out = self.final_layer(input)
            return out, trans_token
