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

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train

class MSDFont_model_eval_Gencase(LatentDiffusion):
    def __init__(self, style_stage_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DiffusionWrapper_MSDFont_model_eval_Gencase(kwargs['unet_config'], kwargs['conditioning_key'])
        self.instantiate_style_stage(style_stage_config)

    def instantiate_style_stage(self, config):
        model = instantiate_from_config(config)
        self.style_stage_model = model

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())

        # #################################################
        # # ignore attn2 parameters, load pretrained diffuser model (e.g., SD released model)
        # ignore_keys = []
        # for k in keys:
        #     if 'attn2' in k:
        #         ignore_keys.append(k)

        # # ignore unet parameters, training from scratch for diffuser
        ignore_keys = []
        for k in keys:
            if 'diffusion_model' in k:
                ignore_keys.append(k)
        print('ignore_keys', ignore_keys)
        # #################################################

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
    
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print('alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        # print('sqrt_one_minus_alphas_cumprod', np.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        ###################################################
        # coefficients for the forward change of DDPM
        ###################################################
        self.register_buffer('forward_chage_posterior_mean_coef1', to_torch(
            (1. - alphas_cumprod_prev ) / (1. - alphas_cumprod_prev + betas)))
        self.register_buffer('forward_chage_posterior_mean_coef2', to_torch(
            betas / (1. - alphas_cumprod_prev + betas)))
        forward_chage_posterior_variance = ( betas * (1. - alphas_cumprod_prev) ) / (1. - alphas_cumprod_prev + betas)
        self.register_buffer('forward_chage_posterior_variance', to_torch(forward_chage_posterior_variance))
        self.register_buffer('forward_chage_posterior_log_variance_clipped', to_torch(np.log(np.maximum(forward_chage_posterior_variance, 1e-20))))
        ###################################################
        # Diffusion Transformation Model for General Case
        print('***********************************************')
        print('Font Transfer Stage for General Case')
        print('self.edit_t1', self.edit_t1)
        print('self.edit_t2', self.edit_t2)
        temp_Ht = torch.ones_like(to_torch(alphas)).numpy()
        sqrt_alphas = np.sqrt(alphas)
        for ii in range(self.edit_t1 + 1, self.edit_t2): # Ht in paper
            temp_Ht[ii] = 1.0 + sqrt_alphas[ii] * temp_Ht[ii-1]
        Ht = temp_Ht
        Ht_prev = np.append(1., Ht[:-1])
        self.register_buffer('forward_chage_Ht', to_torch(Ht)) # Ht in paper
        trans_coeff = self.sqrt_alphas_cumprod[int(self.edit_t2-1)] / Ht[int(self.edit_t2-1)] # Psi in paper
        self.register_buffer('forward_chage_trans_coeff', to_torch(trans_coeff)) # Psi in paper

        forward_chage_posterior_mean_coef1Gencase = (1.0 / (1. - alphas_cumprod)) * ( np.sqrt(alphas) * (1. - alphas_cumprod_prev) * trans_coeff.numpy() + betas * (np.sqrt(alphas_cumprod_prev) - trans_coeff.numpy() * Ht_prev))
        self.register_buffer('forward_chage_posterior_mean_coef1Gencase', to_torch(forward_chage_posterior_mean_coef1Gencase))
        forward_chage_posterior_mean_coef2Gencase = trans_coeff.numpy() * (1.0 / (1. - alphas_cumprod)) * (betas * Ht_prev - np.sqrt(alphas) * (1. - alphas_cumprod_prev))
        self.register_buffer('forward_chage_posterior_mean_coef2Gencase', to_torch(forward_chage_posterior_mean_coef2Gencase))
        ###################################################
        if self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
    
    def get_input_ori(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_input_ori_flat(self, batch, k):
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
    
    def q_sample(self, x_start, c, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        zf = torch.cat(c['zf'], 1)
        temp1 = t >= self.edit_t1
        temp2 = t <= self.edit_t2
        temp3 = temp1 * temp2
        temp3 = temp3.long()
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise - 
                temp3.view(-1,1,1,1) * (x_start-zf) * extract_into_tensor(self.forward_chage_Ht, t, x_start.shape) * self.forward_chage_trans_coeff)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, c=cond, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        target = x_start

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
    
    def shared_step(self, batch, **kwargs):
        x, zf = self.get_input(batch, self.first_stage_key)

        xcs = self.get_input_ori_flat(batch, "style_imgs")
        xcs = xcs.to(self.device) 
        _, zcs = self.style_stage_model.encode(xcs)
        nbs, _, _, _ = zf.shape
        _, nc, nh, nw = zcs.shape
        zcs = zcs.view(nbs, 3, nc, nh, nw)
        zcs = torch.mean(zcs, dim=1)

        xcf = self.get_input_ori(batch, "grey_char_img")
        zcf, _ = self.style_stage_model.encode(xcf)

        c = dict(zf=[zf], zcs=[zcs], zcf=[zcf])

        loss = self(x, c)
        return loss
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(self.edit_t1, self.edit_t2, (x.shape[0],), device=self.device).long() 

        return self.p_losses(x, c, t, *args, **kwargs)
    
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
       # nbs, _, _, _ = zf.shape
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
    
    def q_posterior(self, x_start, x_t, z2, t):
        # print('t', t)
        temp_time = t[0]
        if temp_time >= self.edit_t2:
            posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * z2 +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
                )
           
        elif temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            posterior_mean = (
                extract_into_tensor(self.forward_chage_posterior_mean_coef2Gencase, t, x_t.shape) * z2 + 
                extract_into_tensor(self.forward_chage_posterior_mean_coef1Gencase, t, x_t.shape) * x_start + 
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
                )
            
        else:
            posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )


        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, rec_model, x, cond1, cond2, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        temp_time = t[0]


        z2 = torch.cat(cond1['zf'], 1)

        if temp_time < self.edit_t2 and temp_time >= self.edit_t1:
            model_out = self.apply_model(x, t_in, cond2, return_ids=return_codebook_ids)
        elif temp_time >= self.edit_t2:
            model_out = z2
        else:
            model_out = rec_model.apply_model(x, t_in, cond1, return_ids=return_codebook_ids)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, z2=z2, t=t)
        
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, rec_model, x, cond1, cond2, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(rec_model=rec_model, x=x, cond1=cond1, cond2=cond2, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def progressive_denoising(self, rec_model, cond1, cond2, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            img, x0_partial = self.p_sample(rec_model, img, cond1, cond2, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates
    
class DiffusionWrapper_MSDFont_model_eval_Gencase(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, cond):
        zcs = torch.cat(cond['zcs'], 1)
        zcf = torch.cat(cond['zcf'], 1)
        bs, nc, nh, nw = zcs.shape
        zcs = zcs.view(bs, nc, -1).permute(0,2,1).contiguous()
        zcf = zcf.view(bs, nc, -1).permute(0,2,1).contiguous()
        cc= torch.cat([zcs, zcf], dim=1)
        out = self.diffusion_model(x, t, context=cc)

        return out