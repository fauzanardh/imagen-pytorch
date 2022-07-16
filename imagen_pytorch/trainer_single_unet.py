from math import sqrt
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import contextmanager
from einops import rearrange, reduce, repeat
from einops_exts import check_shape
from ema_pytorch import EMA
import pytorch_warmup as warmup

from bitsandbytes.optim import Adam8bit, AdamW8bit, GlobalOptimManager
from transformers.optimization import Adafactor

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from imagen_pytorch.imagen_pytorch import (
    GaussianDiffusion,
    GaussianDiffusionContinuousTimes,
    Unet,
    cast_tuple,
    eval_decorator,
    identity,
    maybe,
    normalize_neg_one_to_one,
    right_pad_dims_to,
    unnormalize_zero_to_one,
    resize_image_to,
)
from imagen_pytorch.elucidated_imagen import (
    log as elu_log,
    Hparams,
)
from imagen_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim, t5_encode_text
from imagen_pytorch.imagen_pytorch import default
from imagen_pytorch.trainer import (
    exists,
    nullcontext,
    groupby_prefix_and_trim,
    cast_torch_tensor,
    split_args_and_kwargs,
    imagen_sample_in_chunks,
)
from imagen_pytorch.version import __version__
from imagen_pytorch.data import cycle


class SingleUnet(nn.Module):
    def __init__(
        self,
        unet,
        unet_num,
        *,
        image_size,
        text_encoder_name=DEFAULT_T5_NAME,
        text_embed_dim=None,
        channels=3,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        noise_schedule="cosine",
        pred_objective="noise",
        lowres_noise_schedule="linear",
        lowres_sample_noise_level=0.2,
        per_sample_random_aug_noise_level=False,
        condition_on_text=True,
        auto_normalize_img=True,
        continuous_times=True,
        p2_loss_weight_gamma=0.5,
        p2_loss_weight_k=1,
        dynamic_thresholding=True,
        dynamic_thresholding_percentile=0.9,
    ):
        super().__init__()

        assert isinstance(unet, (Unet)), "unet must be an instance of Unet"

        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "huber":
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"loss_type {loss_type} not implemented")

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams
        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels
        self.channels = channels

        # construct noise scheduler
        noise_scheduler_class = (
            GaussianDiffusion
            if not continuous_times
            else GaussianDiffusionContinuousTimes
        )
        # default to cosine noise schedule for first and second unet, and linear for the rest
        noise_schedule = default(noise_schedule, "cosine") if unet_num < 3 else "linear"
        self.noise_scheduler = noise_scheduler_class(
            noise_schedule=noise_schedule, timesteps=timesteps
        )

        # lowres augmentation noise schedule
        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(
            noise_schedule=lowres_noise_schedule
        )

        # ddpm objective
        self.pred_objective = pred_objective

        # text encoder
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(
            text_embed_dim, lambda: get_encoded_dim(text_encoder_name)
        )

        # construct unet
        unet_device = next(unet.parameters()).device
        self.unet = unet.cast_model_parameters(
            lowres_cond=not (unet_num == 0),
            cond_on_text=self.condition_on_text,
            text_embed_dim=self.text_embed_dim if self.condition_on_text else None,
            channels=self.channels,
            channels_out=self.channels,
            learned_sinu_pos_emb=continuous_times,
        ).to(unet_device)
        self.image_size = image_size
        self.sample_channels = self.channels

        # cascading ddpm related stuff
        lowres_condition = self.unet.lowres_cond
        assert (
            lowres_condition is False if unet_num == 0 else True
        ), "lowres_condition must be False for the first unet and the rest must have lowres_condition True"
        self.lowres_condition = lowres_condition

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = self.cond_drop_prob > 0.0

        # normalize and unnormalize image functions
        self.normalize_img = (
            normalize_neg_one_to_one if auto_normalize_img else identity
        )
        self.unnormalize_img = (
            unnormalize_zero_to_one if auto_normalize_img else identity
        )
        self.input_image_range = (0.0 if auto_normalize_img else -1.0, 1.0)

        # dynamic thresholding
        self.dynamic_thresholding = dynamic_thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # p2 loss weight
        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        assert (
            self.p2_loss_weight_gamma <= 2
        ), "in the paper, gamma greater than 2 is harmful"

        # one temp parameter for keeping track of device
        self.register_buffer("_temp", torch.Tensor([0.0]), persistent=False)

        # default to device of unet passed in
        self.to(unet_device)

    @property
    def device(self):
        return self._temp.device

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        cond_scale=1.0,
        model_output=None,
        t_next=None,
        pred_objective="noise",
        dynamic_threshold=True,
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance"

        pred = default(
            model_output,
            lambda: unet.forward_with_cond_scale(
                x,
                noise_scheduler.get_condition(t),
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_images=cond_images,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=self.lowres_noise_schedule.get_condition(
                    lowres_noise_times
                ),
            ),
        )

        if pred_objective == "noise":
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        elif pred_objective == "x_start":
            x_start = pred
        else:
            raise ValueError(f"unknown pred_objective {pred_objective}")

        if dynamic_threshold:
            s = torch.quantile(
                rearrange(x_start, "b ... -> b (...)").abs(),
                q=self.dynamic_thresholding_percentile,
                dim=-1,
            )
            s.clamp_(min=1.0)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1.0, 1.0)

        return noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t, t_next=t_next)

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        t_next=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        cond_scale=1.0,
        lowres_cond_img=None,
        lowres_noise_times=None,
        pred_objective="noise",
        dynamic_threshold=True,
    ):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            t_next=t_next,
            noise_scheduler=noise_scheduler,
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_times=lowres_noise_times,
            pred_objective=pred_objective,
            dynamic_threshold=dynamic_threshold,
        )
        noise = torch.randn_like(x)
        is_last_sampling_timestep = (
            (t_next == 0)
            if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes)
            else (t == 0)
        )
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(
            b, *((1,) * (len(x.size()) - 1))
        )
        return model_mean + nonzero_mask * (model_log_variance / 2).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        cond_scale=1,
        pred_objective="noise",
        dynamic_threshold=True,
    ):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device=device)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device=device)
        for times, times_next in tqdm(
            timesteps, desc="sampling loop time step", total=len(timesteps)
        ):
            img = self.p_sample(
                unet,
                img,
                times,
                t_next=times_next,
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_images=cond_images,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=lowres_noise_times,
                noise_scheduler=noise_scheduler,
                pred_objective=pred_objective,
                dynamic_threshold=dynamic_threshold,
            )

        img.clamp_(-1.0, 1.0)
        return self.unnormalize_img(img)

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts=None,
        text_masks=None,
        text_embeds=None,
        cond_images=None,
        lowres_cond_images=None,
        batch_size=1,
        cond_scale=1.0,
        lowres_sample_noise_level=None,
        return_pil_images=False,
        device=None,
    ):
        device = default(device, lambda: next(self.parameters()).device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(
                lambda t: t.to(device), (text_embeds, text_masks)
            )

        if not self.unconditional:
            batch_size = text_embeds.size(0)

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), "text_embeds must be provided if condition_on_text is True"
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), "text_embeds must not be provided if condition_on_text is False"
        assert not (
            exists(text_embeds) and text_embeds.size(-1) != self.text_embed_dim
        ), f"text_embeds must have shape (batch_size, {self.text_embed_dim})"

        outputs = []

        lowres_sample_noise_level = default(
            lowres_sample_noise_level, self.lowres_sample_noise_level
        )

        lowres_cond_img = lowres_noise_times = None
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        if self.unet.lowres_cond:
            assert (
                lowres_cond_images is not None
            ), "lowres_cond_images must be passed in if lowres_cond is True"
            lowres_noise_times = self.lowres_noise_schedule.get_times(
                batch_size, lowres_sample_noise_level, device=device
            )
            lowres_cond_img = resize_image_to(lowres_cond_images, self.image_size)
            lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img,
                t=lowres_noise_times,
                noise=torch.randn_like(lowres_cond_img),
            )

        img = self.p_sample_loop(
            self.unet,
            shape,
            text_embeds=text_embeds,
            text_mask=text_masks,
            cond_images=cond_images,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_times=lowres_noise_times,
            noise_scheduler=self.noise_scheduler,
            pred_objective=self.pred_objective,
            dynamic_threshold=self.dynamic_thresholding,
        )
        outputs.append(img)

        if not return_pil_images:
            return outputs

        pil_images = list(
            map(lambda img: list(map(T.ToPILImage(), img.unbind(dim=0))), outputs)
        )
        return pil_images

    def p_losses(
        self,
        unet,
        x_start,
        times,
        *,
        noise_scheduler,
        lowres_cond_img=None,
        lowres_aug_times=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        noise=None,
        times_next=None,
        pred_objective="noise",
        p2_loss_weight_gamma=0.0,
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]
        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t
        x_noisy, log_snr = noise_scheduler.q_sample(
            x_start=x_start,
            t=times,
            noise=noise,
        )

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img,
                t=lowres_aug_times,
                noise=torch.randn_like(lowres_cond_img),
            )

        pred = unet(
            x_noisy,
            noise_scheduler.get_condition(times),
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(
                lowres_aug_times
            ),
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob,
        )

        # prediction objective
        if pred_objective == "noise":
            target = noise
        elif pred_objective == "x_start":
            target = x_start
        else:
            raise ValueError("pred_objective must be one of 'noise', 'x_start'")

        # losses
        losses = self.loss_fn(pred, target, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        # p2 loss reweighting
        if p2_loss_weight_gamma > 0.0:
            loss_weight = (
                self.p2_loss_weight_k + log_snr.exp()
            ) ** -p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()

    def forward(
        self,
        images,
        texts=None,
        text_embeds=None,
        text_masks=None,
        cond_images=None,
    ):
        unet = self.unet

        noise_scheduler = self.noise_scheduler
        p2_loss_weight_gamma = self.p2_loss_weight_gamma
        pred_objective = self.pred_objective
        target_image_size = self.image_size
        prev_image_size = self.image_size
        b, _, h, w, device = *images.size(), images.device  # type: ignore

        check_shape(images, "b c h w", c=self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = noise_scheduler.sample_random_times(b, device=device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert len(texts) == len(
                images
            ), "number of text captions does not match up with the number of images given"

            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(
                lambda t: t.to(images.device), (text_embeds, text_masks)
            )

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), "text or text encodings must be passed into decoder if specified"
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), "decoder specified not to be conditioned on text, yet it is presented"
        assert not (
            exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim
        ), f"invalid text embedding dimension being passed in (should be {self.text_embed_dim})"

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(
                images,
                prev_image_size,
                clamp_range=self.input_image_range,
            )
            lowres_cond_img = resize_image_to(
                lowres_cond_img,
                target_image_size,
                clamp_range=self.input_image_range,
            )

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(
                    b, device=device
                )
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(
                    1, device=device
                )
                lowres_aug_times = repeat(lowres_aug_time, "1 -> b", b=b)

        images = resize_image_to(images, target_image_size)
        return self.p_losses(
            unet,
            images,
            times,
            text_embeds=text_embeds,
            text_mask=text_masks,
            cond_images=cond_images,
            noise_scheduler=noise_scheduler,
            lowres_cond_img=lowres_cond_img,
            lowres_aug_times=lowres_aug_times,
            pred_objective=pred_objective,
            p2_loss_weight_gamma=p2_loss_weight_gamma,
        )


class ElucidatedSingleUnet(nn.Module):
    def __init__(
        self,
        unet,
        unet_num,
        *,
        image_size,
        text_encoder_name=DEFAULT_T5_NAME,
        text_embed_dim=None,
        channels=3,
        cond_drop_prob=0.1,
        lowres_sample_noise_level=0.2,
        per_sample_random_aug_noise_level=False,
        condition_on_text=True,
        auto_normalize_img=True,
        dynamic_thresholding=True,
        dynamic_thresholding_percentile=0.9,
        lowres_noise_schedule="linear",
        num_sample_steps=32,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=80,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
    ):
        super().__init__()

        self.unet_num = unet_num

        # conditioning hparams
        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels
        self.channels = channels

        # lowres augmentation noise schedule
        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(
            noise_schedule=lowres_noise_schedule
        )

        # get text encoder
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(
            text_embed_dim, lambda: get_encoded_dim(text_encoder_name)
        )

        # construct unets
        self.unet = unet.cast_model_parameters(
            lowres_cond=unet_num != 0,
            cond_on_text=self.condition_on_text,
            text_embed_dim=self.text_embed_dim if self.condition_on_text else None,
            channels=self.channels,
            channels_out=self.channels,
            learned_sinu_pos_emb=True,
        )

        # unet image sizes
        self.image_size = image_size
        self.sample_channels = self.channels

        # cascading ddpm related stuff
        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.0

        # normalize and unnormalize image functions
        self.normalize_img = (
            normalize_neg_one_to_one if auto_normalize_img else identity
        )
        self.unnormalize_img = (
            unnormalize_zero_to_one if auto_normalize_img else identity
        )
        self.input_image_range = (0.0 if auto_normalize_img else -1.0, 1.0)

        # dynamic thresholding
        self.dynamic_thresholding = dynamic_thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # elucidating parameters
        hparams = [
            num_sample_steps,
            sigma_min,
            sigma_max,
            sigma_data,
            rho,
            P_mean,
            P_std,
            S_churn,
            S_tmin,
            S_tmax,
            S_noise,
        ]

        self.hparams = Hparams(*hparams)

        # one temp parameter for keeping track of device
        self.register_buffer("_temp", torch.tensor([0.0]), persistent=False)

        # default to device of unets passed in
        self.to(next(self.unet.parameters()).device)

    @property
    def device(self):
        return self._temp.device

    # dynamic thresholding
    def threshold_x_start(self, x_start, dynamic_threshold=True):
        if not dynamic_threshold:
            return x_start.clamp(-1.0, 1.0)

        s = torch.quantile(
            rearrange(x_start, "b ... -> b (...)").abs(),
            self.dynamic_thresholding_percentile,
            dim=-1,
        )

        s.clamp_(min=1.0)
        s = right_pad_dims_to(x_start, s)
        return x_start.clamp(-s, s) / s

    # derived preconditioning params - Table 1
    def c_skip(self, sigma_data, sigma):
        return (sigma_data**2) / (sigma**2 + sigma_data**2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma**2 + sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return elu_log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper
    def preconditioned_network_forward(
        self,
        unet_forward,
        noised_images,
        sigma,
        *,
        sigma_data,
        clamp=False,
        dynamic_threshold=True,
        **kwargs,
    ):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1 1")

        net_out = unet_forward(
            self.c_in(sigma_data, padded_sigma) * noised_images,
            self.c_noise(sigma),
            **kwargs,
        )
        out = (
            self.c_skip(sigma_data, padded_sigma) * noised_images
            + self.c_out(sigma_data, padded_sigma) * net_out
        )

        if not clamp:
            return out

        return self.threshold_x_start(out, dynamic_threshold)

    # sampling

    # sample schedule
    # equation (5) in the paper
    def sample_schedule(self, num_sample_steps, rho, sigma_min, sigma_max):
        N = num_sample_steps
        inv_rho = 1 / rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            sigma_max**inv_rho
            + steps / (N - 1) * (sigma_min**inv_rho - sigma_max**inv_rho)
        ) ** rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def one_unet_sample(
        self,
        unet,
        shape,
        *,
        unet_number,
        clamp=True,
        dynamic_threshold=True,
        cond_scale=1.0,
        **kwargs,
    ):
        # get specific sampling hyperparameters for unet

        hp = self.hparams

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(
            hp.num_sample_steps, hp.rho, hp.sigma_min, hp.sigma_max
        )

        gammas = torch.where(
            (sigmas >= hp.S_tmin) & (sigmas <= hp.S_tmax),
            min(hp.S_churn / hp.num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=self.device)

        # unet kwargs

        unet_kwargs = dict(
            sigma_data=hp.sigma_data,
            clamp=clamp,
            dynamic_threshold=dynamic_threshold,
            cond_scale=cond_scale,
            **kwargs,
        )

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas, desc="sampling time step"
        ):
            sigma, sigma_next, gamma = map(
                lambda t: t.item(), (sigma, sigma_next, gamma)
            )

            eps = hp.S_noise * torch.randn(
                shape, device=self.device
            )  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self.preconditioned_network_forward(
                unet.forward_with_cond_scale, images_hat, sigma_hat, **unet_kwargs
            )

            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    unet.forward_with_cond_scale, images_next, sigma_next, **unet_kwargs
                )

                denoised_prime_over_sigma = (
                    images_next - model_output_next
                ) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            images = images_next

        images = images.clamp(-1.0, 1.0)
        return self.unnormalize_img(images)

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts=None,
        text_masks=None,
        text_embeds=None,
        cond_images=None,
        lowres_cond_images=None,
        batch_size=1,
        cond_scale=1.0,
        lowres_sample_noise_level=None,
        return_pil_images=False,
        device=None,
    ):
        device = default(device, lambda: next(self.parameters()).device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(
                lambda t: t.to(device), (text_embeds, text_masks)
            )

        if not self.unconditional:
            batch_size = text_embeds.shape[0]

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), "text or text encodings must be passed into imagen if specified"
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), "imagen specified not to be conditioned on text, yet it is presented"
        assert not (
            exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim
        ), f"invalid text embedding dimension being passed in (should be {self.text_embed_dim})"

        outputs = []
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(
            lowres_sample_noise_level, self.lowres_sample_noise_level
        )

        lowres_cond_img = lowres_noise_times = None
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        if self.unet.lowres_cond:
            assert (
                lowres_cond_images is not None
            ), "lowres_cond_images must be passed in if lowres_cond is True"
            lowres_noise_times = self.lowres_noise_schedule.get_times(
                batch_size, lowres_sample_noise_level, device=device
            )
            lowres_cond_img = resize_image_to(lowres_cond_images, self.image_size)
            lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img,
                t=lowres_noise_times,
                noise=torch.randn_like(lowres_cond_img),
            )

        img = self.one_unet_sample(
            self.unet,
            shape,
            unet_number=self.unet_num,
            text_embeds=text_embeds,
            text_mask=text_masks,
            cond_images=cond_images,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_times=lowres_noise_times,
            dynamic_threshold=self.dynamic_thresholding,
        )
        outputs.append(img)

        if not return_pil_images:
            return outputs

        pil_images = list(
            map(lambda img: list(map(T.ToPILImage(), img.unbind(dim=0))), outputs)
        )
        return pil_images

    # training
    def loss_weight(self, sigma_data, sigma):
        return (sigma**2 + sigma_data**2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, batch_size):
        return (P_mean + P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(
        self,
        images,
        texts=None,
        text_embeds=None,
        text_masks=None,
        unet_number=None,
        cond_images=None,
    ):

        unet = self.unet

        target_image_size = self.image_size
        prev_image_size = None if self.unet_num == 0 else self.image_size
        hp = self.hparams

        batch_size, _, h, w, device, = (  # type: ignore
            *images.shape,
            images.device,
        )

        check_shape(images, "b c h w", c=self.channels)
        assert h >= target_image_size and w >= target_image_size

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert len(texts) == len(
                images
            ), "number of text captions does not match up with the number of images given"

            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(
                lambda t: t.to(images.device), (text_embeds, text_masks)
            )

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), "text or text encodings must be passed into decoder if specified"
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), "decoder specified not to be conditioned on text, yet it is presented"
        assert not (
            exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim
        ), f"invalid text embedding dimension being passed in (should be {self.text_embed_dim})"

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(
                images,
                prev_image_size,
                clamp_range=self.input_image_range,
            )
            lowres_cond_img = resize_image_to(
                lowres_cond_img,
                target_image_size,
                clamp_range=self.input_image_range,
            )

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(
                    batch_size, device=device
                )
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(
                    1, device=device
                )
                lowres_aug_times = repeat(lowres_aug_time, "1 -> b", b=batch_size)

        images = resize_image_to(images, target_image_size)

        # normalize to [-1, 1]
        images = self.normalize_img(images)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3
        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img,
                t=lowres_aug_times,
                noise=torch.randn_like(lowres_cond_img),
            )

        # get the sigmas
        sigmas = self.noise_distribution(hp.P_mean, hp.P_std, batch_size)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1 1")

        # noise
        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        # get prediction
        denoised_images = self.preconditioned_network_forward(
            unet.forward,
            noised_images,
            sigmas,
            sigma_data=hp.sigma_data,
            text_embeds=text_embeds,
            text_mask=text_masks,
            cond_images=cond_images,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(
                lowres_aug_times
            ),
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob,
        )

        # losses
        losses = F.mse_loss(denoised_images, images, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        # loss weighting
        losses = losses * self.loss_weight(hp.sigma_data, sigmas)

        # return average loss
        return losses.mean()


class SingleUnetTrainer(nn.Module):
    locked = False

    def __init__(
        self,
        single_unet,
        use_ema=True,
        lr=1e-4,
        eps=1e-8,
        beta1=0.9,
        beta2=0.99,
        max_grad_norm=None,
        # group_wd_params=True,
        warmup_steps=None,
        cosine_decay_max_steps=None,
        train_dl=None,
        valid_dl=None,
        fp16=False,
        split_batches=True,
        dl_tuple_output_keywords_names=(
            "images",
            "text_embeds",
            "text_masks",
            "cond_images",
        ),
        optimizer="adam",
        t5_encoder_name=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(single_unet, (ElucidatedSingleUnet, SingleUnet))
        ema_kwargs, kwargs = groupby_prefix_and_trim("ema_", kwargs)
        accelerate_kwargs, kwargs = groupby_prefix_and_trim("accelerate_", kwargs)

        self.accelerator = Accelerator(
            **{
                "split_batches": split_batches,
                "mixed_precision": "fp16" if fp16 else "no",
                "kwargs_handlers": [
                    DistributedDataParallelKwargs(find_unused_parameters=True)
                ],
                **accelerate_kwargs,
            }
        )
        SingleUnetTrainer.locked = self.is_distributed

        grad_scaler_enabled = fp16
        self.single_unet = single_unet
        self.t5_encoder_name = t5_encoder_name
        self.verbose = verbose
        self.use_ema = use_ema and self.accelerator.is_main_process
        self.ema_unet = None

        self.train_dl_iter = None
        self.train_dl = None

        self.valid_dl_iter = None
        self.valid_dl = None

        self.dl_tuple_output_keywords_names = dl_tuple_output_keywords_names

        self.add_train_dataloader(train_dl)
        self.add_valid_dataloader(valid_dl)

        if optimizer == "adam":
            optim = Adam(
                single_unet.unet.parameters(),
                lr=lr,
                eps=eps,
                betas=(beta1, beta2),
                **kwargs,
            )
        elif optimizer == "adamw":
            optim = AdamW(
                single_unet.unet.parameters(),
                lr=lr,
                eps=eps,
                betas=(beta1, beta2),
                **kwargs,
            )
        elif optimizer == "adam8bit":
            optim = Adam8bit(
                single_unet.unet.parameters(),
                lr=lr,
                eps=eps,
                betas=(beta1, beta2),
                **kwargs,
            )
            for module in single_unet.unet.modules():
                if isinstance(module, nn.Embedding):
                    GlobalOptimManager.get_instance().register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
        elif optimizer == "adamw8bit":
            optim = AdamW8bit(
                single_unet.unet.parameters(),
                lr=lr,
                eps=eps,
                betas=(beta1, beta2),
                **kwargs,
            )
            for module in single_unet.unet.modules():
                if isinstance(module, nn.Embedding):
                    GlobalOptimManager.get_instance().register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
        elif optimizer == "adafactor":
            optim = Adafactor(
                single_unet.unet.parameters(),
                lr=lr,
                eps=eps,
                beta1=beta1,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"optimizer {optimizer} not implemented")

        self.optim = optim

        if self.use_ema:
            self.ema_unet = EMA(single_unet.unet, **ema_kwargs)

        scaler = GradScaler(enabled=grad_scaler_enabled)
        self.scaler = scaler

        scheduler = warmup_scheduler = None
        if exists(cosine_decay_max_steps):
            scheduler = CosineAnnealingLR(optim, T_max=cosine_decay_max_steps)
        if exists(warmup_steps):
            warmup_scheduler = warmup.LinearWarmup(optim, warmup_steps)

        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler

        self.max_grad_norm = max_grad_norm
        self.register_buffer("steps", torch.Tensor([0]))
        self.register_buffer("_temp", torch.tensor([0.0]), persistent=False)

        self.to(next(single_unet.parameters()).device)

    @property
    def device(self):
        return self._temp.device

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    def add_train_dataloader(self, dl=None):
        if not exists(dl):
            return
        assert not exists(self.train_dl), "train dl already exists"
        self.train_dl = self.accelerator.prepare(dl)

    def add_valid_dataloader(self, dl=None):
        if not exists(dl):
            return
        assert not exists(self.valid_dl), "valid dl already exists"
        self.valid_dl = self.accelerator.prepare(dl)

    def create_train_iter(self):
        assert exists(self.train_dl), "train dl does not exist"
        if exists(self.train_dl_iter):
            return
        self.train_dl_iter = cycle(self.train_dl)

    def create_valid_iter(self):
        assert exists(self.valid_dl), "valid dl does not exist"
        if exists(self.valid_dl_iter):
            return
        self.valid_dl_iter = cycle(self.valid_dl)

    def train_step(self, **kwargs):
        self.create_train_iter()
        loss = self.step_with_dl_iter(self.train_dl_iter, **kwargs)
        self.update()
        return loss

    def save(self, path, overwrite=True, **kwargs):
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_local_main_process:
            return

        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_obj = dict(
            model=self.single_unet.state_dict(),
            version=__version__,
            step=self.steps.cpu(),
            **kwargs,
        )

        if exists(self.scheduler):
            save_obj["scheduler"] = self.scheduler.state_dict()
        if exists(self.warmup_scheduler):
            save_obj["warmup"] = self.warmup_scheduler.state_dict()

        save_obj["scaler"] = self.scaler.state_dict()
        save_obj["optimizer"] = self.optim.state_dict()

        if self.use_ema:
            save_obj["ema"] = self.ema_unet.state_dict()

        torch.save(save_obj, str(path))

    def load(self, path, only_model=False, strict=True):
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path))

        self.single_unet.load_state_dict(loaded_obj["model"], strict=strict)

        if only_model:
            return loaded_obj

        if exists(self.scheduler):
            self.scheduler.load_state_dict(loaded_obj["scheduler"])
        if exists(self.warmup_scheduler):
            self.warmup_scheduler.load_state_dict(loaded_obj["warmup"])

        self.scaler.load_state_dict(loaded_obj["scaler"])
        self.optim.load_state_dict(loaded_obj["optimizer"])

        if self.use_ema:
            assert "ema" in loaded_obj
            self.ema_unet.load_state_dict(loaded_obj["ema"])

        return loaded_obj

    @torch.no_grad()
    @eval_decorator
    def valid_step(self, **kwargs):
        self.create_valid_iter()
        context = self.use_ema_unet if self.use_ema else nullcontext

        with context():
            loss = self.step_with_dl_iter(self.valid_dl_iter, **kwargs)
        return loss

    def step_with_dl_iter(self, dl_iter, **kwargs):
        dl_tuple_output = cast_tuple(next(dl_iter))
        model_input = dict(
            list(zip(self.dl_tuple_output_keywords_names, dl_tuple_output))
        )
        loss = self.forward(**{**kwargs, **model_input})
        return loss

    @torch.no_grad()
    @contextmanager
    def use_ema_unet(self):
        if not self.use_ema:
            output = yield
            return output

        self.single_unet.eval()
        trainable_unet = self.single_unet.unet
        self.single_unet.unet = self.ema_unet.ema_model

        output = yield

        self.single_unet.unet = trainable_unet
        self.ema_unet.restore_ema_model_device()
        return output

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def encode_text(self, text, **kwargs):
        return t5_encode_text(text, name=self.t5_encoder_name, **kwargs)

    def update(self):
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(
                self.single_unet.unet.parameters(), self.max_grad_norm
            )

        self.optim.step()
        self.optim.zero_grad()

        if self.use_ema:
            self.ema_unet.update()

        maybe_warmup_context = (
            nullcontext()
            if not exists(self.warmup_scheduler)
            else self.warmup_scheduler.dampening()
        )
        with maybe_warmup_context:
            if (
                exists(self.scheduler)
                and not self.accelerator.optimizer_step_was_skipped
            ):
                self.scheduler.step()

        self.steps += 1

    @torch.no_grad()
    @cast_torch_tensor
    @imagen_sample_in_chunks
    def sample(self, *args, **kwargs):
        context = self.use_ema_unet if self.use_ema else nullcontext

        with context():
            output = self.single_unet.sample(*args, **kwargs)

        return output

    @cast_torch_tensor
    def forward(
        self,
        *args,
        max_batch_size=None,
        **kwargs,
    ):
        total_loss = 0.0
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(
            *args,
            split_size=max_batch_size,
            **kwargs,
        ):
            with self.accelerator.autocast():
                loss = self.single_unet(
                    *chunked_args,
                    **chunked_kwargs,
                )
                loss = loss * chunk_size_frac
            total_loss += loss.item()

            if self.training:
                self.accelerator.backward(loss)

        return total_loss
