import torch
from torch import nn
from torch.optim import SGD, Adagrad

from transformers import PreTrainedModel, PretrainedConfig
from .modules import Block
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from .perplexity import calculate_perplexity


class VQAdapterConfig(PretrainedConfig):
    def __init__(
        self,
        # vq args
        # All of these have 'vq_' appended to the start
        vq_codebook_size: int = 32,
        vq_codebook_dim: int = 32,
        vq_heads: int = 32,
        vq_separate_codebook_per_head: bool = True,
        vq_decay: float = 0.85,
        vq_eps: float = 1e-5,
        vq_kmeans_init: bool = False,
        vq_kmeans_iters: int = 20,
        vq_sync_kmeans: bool = True,
        vq_use_cosine_sim: bool = False,
        vq_threshold_ema_dead_code: int = 0,
        vq_channel_last: bool = True,
        vq_accept_image_fmap: bool = False,
        vq_commitment_weight: float = 1.0,
        vq_commitment_use_cross_entropy_loss: bool = False,
        vq_orthogonal_reg_weight: float = 0.0,
        vq_orthogonal_reg_active_codes_only: bool = False,
        vq_orthogonal_reg_max_codes: bool = None,
        vq_stochastic_sample_codes: bool = True,
        vq_sample_codebook_temp: float = 1.0,
        vq_straight_through: bool = False,
        vq_reinmax: bool = False,
        # using reinmax for improved straight-through, assuming straight through helps at all
        vq_sync_codebook: bool = False,
        vq_sync_affine_param: bool = False,
        vq_ema_update: bool = True,
        vq_learnable_codebook: bool = False,
        vq_affine_param: bool = False,
        vq_affine_param_batch_decay: float = 0.99,
        vq_affine_param_codebook_decay: float = 0.9,
        # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
        vq_sync_update_v: float = 0.0,

        # codebook optimizer
        codebook_lr: float = 10.,

        # rq_specific args
        rq_quantize_dropout=False,
        rq_quantize_dropout_cutoff_index=0,
        rq_quantize_dropout_multiple_of=1,

        # nn args
        is_rq: bool = True,
        mlp_dim: int = 1028,
        mlp_hidden_dim: int = 512,
        mlp_layers: int = 1,
        # Default clip dim for L models
        clip_dim: int = 768,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vq_codebook_size = vq_codebook_size
        self.vq_codebook_dim = vq_codebook_dim
        self.vq_heads = vq_heads
        self.vq_separate_codebook_per_head = vq_separate_codebook_per_head
        self.vq_decay = vq_decay
        self.vq_eps = vq_eps
        self.vq_kmeans_init = vq_kmeans_init
        self.vq_kmeans_iters = vq_kmeans_iters
        self.vq_sync_kmeans = vq_sync_kmeans
        self.vq_use_cosine_sim = vq_use_cosine_sim
        self.vq_threshold_ema_dead_code = vq_threshold_ema_dead_code
        self.vq_channel_last = vq_channel_last
        self.vq_accept_image_fmap = vq_accept_image_fmap
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_commitment_use_cross_entropy_loss = vq_commitment_use_cross_entropy_loss
        self.vq_orthogonal_reg_weight = vq_orthogonal_reg_weight
        self.vq_orthogonal_reg_active_codes_only = vq_orthogonal_reg_active_codes_only
        self.vq_orthogonal_reg_max_codes = vq_orthogonal_reg_max_codes
        self.vq_stochastic_sample_codes = vq_stochastic_sample_codes
        self.vq_sample_codebook_temp = vq_sample_codebook_temp
        self.vq_straight_through = vq_straight_through
        self.vq_reinmax = vq_reinmax
        self.vq_sync_codebook = vq_sync_codebook
        self.vq_sync_affine_param = vq_sync_affine_param
        self.vq_ema_update = vq_ema_update
        self.vq_learnable_codebook = vq_learnable_codebook
        self.vq_affine_param = vq_affine_param
        self.vq_affine_param_batch_decay = vq_affine_param_batch_decay
        self.vq_affine_param_codebook_decay = vq_affine_param_codebook_decay
        self.vq_sync_update_v = vq_sync_update_v

        self.codebook_lr=codebook_lr

        self.rq_quantize_dropout=rq_quantize_dropout
        self.rq_quantize_dropout_cutoff_index=rq_quantize_dropout_cutoff_index
        self.rq_quantize_dropout_multiple_of=rq_quantize_dropout_multiple_of

        self.is_rq = is_rq
        self.mlp_dim = mlp_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_layers = mlp_layers
        self.clip_dim = clip_dim


class VQAdapterModel(PreTrainedModel):
    config_class = VQAdapterConfig

    def __init__(self, config: VQAdapterConfig):
        super().__init__(config)

        quantizer_args = {
            k.removeprefix("vq_"): v
            for k, v in config.to_dict().items()
            if k.startswith("vq_")
        }

        #if quantizer_args['learnable_codebook']:
            #quantizer_args['in_place_codebook_optimizer'] = lambda *args, **kwargs: Adagrad(*args, lr=config.codebook_lr, **kwargs)

        quantizer_args["dim"] = config.clip_dim
        if config.is_rq:
            rq_args = {
                k.removeprefix("rq_"): v
                for k, v in config.to_dict().items()
                if k.startswith("rq_")
            }
            quantizer_args.update(rq_args)
            quantizer_args["heads"] = 1
            quantizer_args["num_quantizers"] = config.vq_heads
            self.vq = ResidualVQ(**quantizer_args)
        else:
            self.vq = VectorQuantize(**quantizer_args)

        self.in_feature_net = nn.Sequential(
            # input is assumed to an already normalized clip embedding
            nn.Linear(config.clip_dim, config.mlp_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(config.mlp_dim),
            *[
                Block(config.mlp_dim, config.mlp_hidden_dim)
                for _ in range(config.mlp_layers)
            ],
            nn.Linear(config.mlp_dim, config.clip_dim, bias=False),
            # normalize before passing to VQ?
            # nn.GELU(),
            # nn.LayerNorm(args.clip_dim),
        )

        self.out_feature_net = nn.Identity()

    def decode(self, codes: torch.LongTensor):
        z = self.vq.get_codes_from_indices(codes)
        z = self.vq.project_out(z)
        return z

    def _init_weights(self, _):
        pass

    def forward(self, z, return_perplexity=False):
        """
        z: B by D
        """
        z = self.in_feature_net(z)
        z, codes, loss = self.vq(z.unsqueeze(1))
        loss = loss.mean()
        z = z.squeeze(1)
        codes = codes.squeeze(1)
        if return_perplexity:
            perplexity = calculate_perplexity(codes, self.config.vq_codebook_size)
        else:
            perplexity = None
        z = self.out_feature_net(z)

        return dict(z=z, codes=codes, perplexity=perplexity, loss=loss)
