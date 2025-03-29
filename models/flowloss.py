import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper


from diffusion import DPMS, FlowEuler, Scheduler

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "logit_normal_trigflow":
        sigma = torch.randn(batch_size, device="cpu")
        sigma = (sigma * logit_std + logit_mean).exp()
        u = torch.atan(sigma / 0.5)  # TODO: 0.5 should be a hyper-parameter
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u





class FlowLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self,flow_method, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(FlowLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )
        self.flow_method=flow_method
        if not self.flow_method=='linear_flow':
            self.path= AffineProbPath(scheduler=CondOTScheduler())
        else:
            self.train_diffusion=Scheduler(
                                str(1000),#1000
                                noise_schedule='linear_flow',#'linear_flow'
                                predict_flow_v=True,#True
                                learn_sigma=False,#False
                                pred_sigma=False,#False
                                snr=False,#False
                                flow_shift=3.0,#3.0
                                )

    def forward(self, target, z, mask=None):

        if not self.flow_method=='linear_flow':
            t = torch.rand(target.shape[0]).to(target.device)
            x_0 = torch.randn_like(target).to(target.device)

            path_sample = self.path.sample(t=t, x_0=x_0, x_1=target)
            loss = (mask[:,None]*torch.pow((self.net(path_sample.x_t,path_sample.t,z) - path_sample.dx_t), 2)).sum() / mask.sum()
            return loss/target.shape[-1]
        else:
            u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',#'logit_normal'
                    batch_size=target.shape[0],
                    logit_mean=0.0,#0.0
                    logit_std=1.0,#1.0
                    mode_scale=None,  # not used
                )
            t = (u * 1000).long().to(target.device)#
            model_kwargs = dict(c=z)
            loss = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
            loss=loss["loss"]
            if mask is not None:
                loss = (loss * mask).sum() / mask.sum()

            return loss.mean()



    def sample(self, z,uncondition=None, temperature=1.0, cfg=1.0):
        if not self.flow_method=='linear_flow':
            if not cfg == 1.0:
                noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
                noise = torch.cat([noise, noise], dim=0)
                model_kwargs = dict(c=z, cfg_scale=cfg)
                sample_fn = self.net.forward_with_cfg
            else:
                noise = torch.randn(z.shape[0], self.in_channels).cuda()
                model_kwargs = dict(c=z)
                sample_fn = self.net.forward
            class WrappedModel(ModelWrapper):
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                    return self.model.forward_with_cfg(x, t,extras['model_extras']['c'],extras['model_extras']['cfg_scale'])
            wrapped_vf = WrappedModel(self.net)
            solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
            T = torch.linspace(0,1,10)  # sample times
            T = T.to(device=z.device)
            step_size = 0.05
            
            sampled_token_latent = solver.sample(time_grid=T, x_init=noise, method='dopri5', step_size=None, return_intermediates=False,atol=1e-5, rtol=1e-5,model_extras=model_kwargs)
            return sampled_token_latent
        else:
            dpm_solver = DPMS(
                    self.net.forward,
                    condition=z,
                    uncondition=uncondition,
                    cfg_scale=4.5,
                    model_type="flow",
                    schedule="FLOW",
                )
            if not cfg==1.0:
                noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
                noise = torch.cat([noise, noise], dim=0)
            else:
                noise = torch.randn(z.shape[0], self.in_channels).cuda()
            denoised = dpm_solver.sample(
                        noise,
                        steps=20,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=3.0,
                    )
            return denoised


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)
    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        t=t[None].expand(x.shape[0])
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

