import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig
from omegaconf import OmegaConf

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train

# from pytorch_lightning.utilities.seed import isolate_rng
# from torch.nn.parallel.distributed import DistributedDataParallel
# from torch.nn import Module
# import logging
# log = logging.getLogger(__name__)

class MSDFont_train_stage2_rec_model_distri(LatentDiffusion):

    def __init__(self, style_stage_config, trans_model_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)

        self.instantiate_style_stage(style_stage_config)

        self.model = DiffusionWrapper_MSDFont_train_stage2_rec_model_distri(kwargs['unet_config'], kwargs['conditioning_key'])
        
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            print('*****************************************************************************')
            print('load the rec Unet model')
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()
        print('end of load the rec Unet model')
        print('*****************************************************************************')

        # print('self.model', self.model)

        self.instantiate_trans_stage(trans_model_config)
        self.device1 = 'cuda:0'
        self.device2 = 'cuda:1'
        self.trans_stage_model = self.trans_stage_model.to(self.device2)
        print('self.trans_stage_model.device', self.trans_stage_model.device)
        

    def instantiate_style_stage(self, config):
        model = instantiate_from_config(config)
        self.style_stage_model = model

    def instantiate_trans_stage(self, config):
        print("*************************************************************************")
        print('instantiate_trans_stage')
        print(config)
        config_path = config.pop("config_path", None)
        model_path = config.pop("model_path", None)
        trans_stage_config = OmegaConf.load(config_path)
        trans_stage_path = model_path
        # print(trans_stage_config)
        trans_model = instantiate_from_config(trans_stage_config.model)
        sd = torch.load(trans_stage_path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        ignore_keys = []
        for k in keys:
            if 'first_stage_model' in k:
                ignore_keys.append(k)
            if 'cond_stage_model' in k:
                ignore_keys.append(k)
            # if 'style_stage_model' in k:
            #     ignore_keys.append(k)
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # print(sd)
        missing, unexpected = trans_model.load_state_dict(sd, strict=False)
        print(f"Restored from {trans_stage_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")
        # trans_model.load_state_dict(sd, strict=False)
        self.trans_stage_model = trans_model.eval()
        self.trans_stage_model.train = disabled_train
        for param in self.trans_stage_model.parameters():
            param.requires_grad = False

        print('end of instantiate_trans_stage')
        print("*************************************************************************")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(self.style_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
    
    def get_input_ori(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_input_ori_flat(self, batch, k): # rewrite get_input_ori for using few-shot refference imgs
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.flatten(0, 1)
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):
        x = self.get_input_ori(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        xf = self.get_input_ori(batch, "char_img")
        xf = xf.to(self.device) 
        xf_encoder_posterior = self.encode_first_stage(xf)
        zf = self.get_first_stage_encoding(xf_encoder_posterior).detach()
        # bs, c, h, w = zf.shape
        # zf = torch.zeros(bs,1024,h,w).to(self.device) 
        if bs is not None:
            zf = zf[:bs]

        c = zf
        xc = None
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(xc)
        return out
    
    def q_sample(self, x0, c, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
    
    def q_sample_trans(self, x_start, c, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        zf = torch.cat(c['zf'], 1)
        temp1 = t >= self.edit_t1
        temp2 = t <= self.edit_t2
        temp3 = temp1 * temp2
        temp3 = temp3.long()
        return (extract_into_tensor(self.trans_stage_model.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.trans_stage_model.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise - 
                temp3.view(-1,1,1,1) * (x_start-zf) * extract_into_tensor(self.trans_stage_model.forward_chage_Ht, t, x_start.shape) * self.trans_stage_model.forward_chage_trans_coeff)
    
    def p_losses_distrill(self, Tx0, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x0=Tx0, c=cond, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # logvar_t = self.logvar[t].to(self.device)
        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    @torch.no_grad()
    def trans_stage_prediction(self, x, zf, xcs, xcf, **kwargs):
        zf = zf.to(self.device2)
        xcs = xcs.to(self.device2) 
        xcf = xcf.to(self.device2)
        x = x.to(self.device2)
        _, zcs2 = self.trans_stage_model.style_stage_model.encode(xcs)
        nbs, _, _, _ = zf.shape
        _, nc, nh, nw = zcs2.shape
        zcs2 = zcs2.view(nbs, 3, nc, nh, nw)
        zcs2 = torch.mean(zcs2, dim=1)
        # xcf = self.get_input_ori(batch, "grey_char_img")
        zcf2, _ = self.trans_stage_model.style_stage_model.encode(xcf)
        c2 = dict(zf=[zf], zcs=[zcs2], zcf=[zcf2])
        t_trans = torch.randint(0, self.edit_t2, (x.shape[0],), device=self.device2).long()
        t_trans = t_trans.to(self.device2)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample_trans(x_start=x, c=c2, t=t_trans, noise=noise)
        model_output = self.trans_stage_model.apply_model(x_noisy, t_trans, c2)
        pred_z0 = model_output
        pred_z0 = pred_z0.to(self.device1)
        return pred_z0
    
    def shared_step(self, batch, **kwargs):
        x, zf = self.get_input(batch, self.first_stage_key)
        xcs = self.get_input_ori_flat(batch, "style_imgs")
        xcf = self.get_input_ori(batch, "grey_char_img")

        trans_pred_z0 = self.trans_stage_prediction(x, zf, xcs, xcf)

        xcs = xcs.to(self.device) 
        _, zcs = self.style_stage_model.encode(xcs)
        nbs, _, _, _ = zf.shape
        _, nc, nh, nw = zcs.shape
        zcs = zcs.view(nbs, 3, nc, nh, nw)
        zcs = torch.mean(zcs, dim=1)
        zcf, _ = self.style_stage_model.encode(xcf)

        # we just combine them into dict, zf is not used as conditions, 
        # please refer function: DiffusionWrapper_DifFonTrans_2cond_2Enc_rec_stage_MiniUnet for details
        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf]) 
        Tx0 = trans_pred_z0.detach()
        loss = self(Tx0, x, c)
        return loss
    
    def forward(self, Tx0, x, c, *args, **kwargs):
        t = torch.randint(0, self.edit_t1, (x.shape[0],), device=self.device).long()

        return self.p_losses_distrill(Tx0, x, c, t, *args, **kwargs)
    
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        x_recon = self.model(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, zf, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        
        xs = self.get_input_ori_flat(batch, "style_imgs")
        xs = xs.to(self.device) 
        _, zs = self.style_stage_model.encode(xs)
        nbs, nc, nh, nw = zs.shape
        zs = zs.view(int(nbs/3), 3, nc, nh, nw)
        zcs = torch.mean(zs, dim=1)
        if N is not None:
            zcs = zcs[:N]

        xcf = self.get_input_ori(batch, "grey_char_img")
        zcf, _ = self.style_stage_model.encode(xcf)
        if N is not None:
            zcf = zcf[:N]

        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf])

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    

class DiffusionWrapper_MSDFont_train_stage2_rec_model_distri(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        # self.diffusion_model = self.instantiate_diffusion_stage(diff_model_config)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, cond):
        device = x.device

        zcs = torch.cat(cond['zcs'], 1).to(device)
        zcf = torch.cat(cond['zcf'], 1).to(device)
        bs, nc, nh, nw = zcs.shape
        zcs = zcs.view(bs, nc, -1).permute(0,2,1).contiguous()
        zcf = zcf.view(bs, nc, -1).permute(0,2,1).contiguous()
        cc= torch.cat([zcs, zcf], dim=1)

        out = self.diffusion_model(x, t, context=cc)

        return out
