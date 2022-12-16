from pydantic import BaseModel, validator
from typing import List, Optional, Union, Tuple
from enum import Enum

from imagen_pytorch.imagen_pytorch import Imagen, Unet, Unet3D, NullUnet
from imagen_pytorch.trainer import ImagenTrainer
from imagen_pytorch.elucidated_imagen import ElucidatedImagen
from imagen_pytorch.t5_encoder import DEFAULT_T5_NAME


# noise schedule
class NoiseSchedule(Enum):
    cosine = "cosine"
    linear = "linear"


class AllowExtraBaseModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True


# imagen pydantic classes
class NullUnetConfig(BaseModel):
    is_null: bool

    def create(self):
        return NullUnet()


class UnetConfig(AllowExtraBaseModel):
    dim: int
    dim_mults: Union[List[int], Tuple[int]]
    text_embed_dim: int
    cond_dim: int = 128
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

    def create(self):
        return Unet(**self.dict())


class Unet3DConfig(AllowExtraBaseModel):
    dim: int
    dim_mults: Union[List[int], Tuple[int]]
    text_embed_dim: int
    cond_dim: int = 128
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

    def create(self):
        return Unet3D(**self.dict())


class ImagenConfig(AllowExtraBaseModel):
    unets: Union[
        List[Union[UnetConfig, Unet3DConfig, NullUnetConfig]],
        Tuple[Union[UnetConfig, Unet3DConfig, NullUnetConfig]],
    ]
    image_sizes: Union[List[int], Tuple[int]]
    video: bool = False
    timesteps: Union[int, Union[List[int], Tuple[int]]] = 1000
    noise_schedules: Union[
        str,
        Union[List[str], Tuple[str]],
        NoiseSchedule,
        Union[List[NoiseSchedule], Tuple[NoiseSchedule]],
    ] = "cosine"
    text_encoder_name: str = DEFAULT_T5_NAME
    channels: int = 3
    loss_type: str = "l2"
    cond_drop_prob: float = 0.5

    @validator("image_sizes")
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get("unets")
        if len(image_sizes) != len(unets):
            raise ValueError(
                f"image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}"
            )
        return image_sizes

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop("unets")
        is_video = decoder_kwargs.pop("video", False)

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = Imagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen


class ElucidatedImagenConfig(AllowExtraBaseModel):
    unets: Union[
        List[Union[UnetConfig, Unet3DConfig, NullUnetConfig]],
        Tuple[Union[UnetConfig, Unet3DConfig, NullUnetConfig]],
    ]
    image_sizes: Union[
        List[Union[List[int], Tuple[int]]], Tuple[Union[List[int], Tuple[int]]]
    ]
    video: bool = False
    text_encoder_type: str = "t5"
    text_encoder_name: Optional[str] = None
    channels: int = 3
    cond_drop_prob: float = 0.5
    num_sample_steps: Union[int, Union[List[int], Tuple[int]]] = 32
    sigma_min: Union[float, Union[List[float], Tuple[float]]] = 0.002
    sigma_max: Union[int, Union[List[int], Tuple[int]]] = 80
    sigma_data: Union[float, Union[List[float], Tuple[float]]] = 0.5
    rho: Union[int, Union[List[int], Tuple[int]]] = 7
    P_mean: Union[float, Union[List[float], Tuple[float]]] = -1.2
    P_std: Union[float, Union[List[float], Tuple[float]]] = 1.2
    S_churn: Union[int, Union[List[int], Tuple[int]]] = 80
    S_tmin: Union[float, Union[List[float], Tuple[float]]] = 0.05
    S_tmax: Union[int, Union[List[int], Tuple[int]]] = 50
    S_noise: Union[float, Union[List[float], Tuple[float]]] = 1.003

    @validator("image_sizes")
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get("unets")
        if len(image_sizes) != len(unets):
            raise ValueError(
                f"image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}"
            )
        return image_sizes

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop("unets")
        is_video = decoder_kwargs.pop("video", False)

        unet_klass = Unet3D if is_video else Unet

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = ElucidatedImagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen


class ImagenTrainerConfig(AllowExtraBaseModel):
    imagen: dict
    elucidated: bool = False
    video: bool = False
    use_ema: bool = True
    lr: Union[float, Union[List[float], Tuple[float]]] = 1e-4
    eps: Union[float, Union[List[float], Tuple[float]]] = 1e-8
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: Optional[float] = None
    group_wd_params: bool = True
    warmup_steps: Union[
        Optional[int], Union[List[Optional[int]], Tuple[Optional[int]]]
    ] = None
    cosine_decay_max_steps: Union[
        Optional[int], Union[List[Optional[int]], Tuple[Optional[int]]]
    ] = None

    def create(self):
        trainer_kwargs = self.dict()

        imagen_config = trainer_kwargs.pop("imagen")
        elucidated = trainer_kwargs.pop("elucidated")
        is_video = trainer_kwargs.pop("video")

        imagen_config_klass = ElucidatedImagenConfig if elucidated else ImagenConfig
        imagen = imagen_config_klass(**{**imagen_config, "video": is_video}).create()

        return ImagenTrainer(imagen, **trainer_kwargs)
