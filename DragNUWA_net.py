from utils import *

#### SVD
from dragnuwa.svd.modules.diffusionmodules.video_model_flow import VideoUNet_flow, VideoResBlock_Embed
from dragnuwa.svd.modules.diffusionmodules.denoiser import Denoiser
from dragnuwa.svd.modules.diffusionmodules.denoiser_scaling import VScalingWithEDMcNoise
from dragnuwa.svd.modules.encoders.modules import *
from dragnuwa.svd.models.autoencoder import AutoencodingEngine
from dragnuwa.svd.modules.diffusionmodules.wrappers import OpenAIWrapper
from dragnuwa.svd.modules.diffusionmodules.sampling import EulerEDMSampler

from dragnuwa.lora import inject_trainable_lora, inject_trainable_lora_extended, extract_lora_ups_down, _find_modules

def get_gaussian_kernel(kernel_size, sigma, channels):
    print('parameters of gaussian kernel: kernel_size: {}, sigma: {}, channels: {}'.format(kernel_size, sigma, channels))
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

def inject_lora(use_lora, model, replace_modules, is_extended=False, dropout=0.0, r=16):
    injector = (
        inject_trainable_lora if not is_extended
        else
        inject_trainable_lora_extended
    )

    params = None
    negation = None

    if use_lora:
        REPLACE_MODULES = replace_modules
        injector_args = {
            "model": model,
            "target_replace_module": REPLACE_MODULES,
            "r": r
        }
        if not is_extended: injector_args['dropout_p'] = dropout

        params, negation = injector(**injector_args)
        for _up, _down in extract_lora_ups_down(
                model,
                target_replace_module=REPLACE_MODULES):

            if all(x is not None for x in [_up, _down]):
                print(f"Lora successfully injected into {model.__class__.__name__}.")

            break

    return params, negation

class Args:
    ### basic
    fps = 4
    height = 320
    width = 576

    ### lora
    unet_lora_rank = 32

    ### gaussian filter parameters
    kernel_size = 199
    sigma = 20

    # model
    denoiser_config = {
        'scaling_config':{
            'target': 'dragnuwa.svd.modules.diffusionmodules.denoiser_scaling.VScalingWithEDMcNoise',
        }
    }

    network_config = {
        'adm_in_channels': 768, 'num_classes': 'sequential', 'use_checkpoint': True, 'in_channels': 8, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_head_channels': 64, 'use_linear_in_transformer': True, 'transformer_depth': 1, 'context_dim': 1024, 'spatial_transformer_attn_type': 'softmax-xformers', 'extra_ff_mix_layer': True, 'use_spatial_context': True, 'merge_strategy': 'learned_with_images', 'video_kernel_size': [3, 1, 1], 'flow_dim_scale': 1,  
    }

    conditioner_emb_models = [
        {'is_trainable': False,
         'input_key': 'cond_frames_without_noise',  # crossattn
         'ucg_rate': 0.1,
         'target': 'dragnuwa.svd.modules.encoders.modules.FrozenOpenCLIPImagePredictionEmbedder',
         'params':{
            'n_cond_frames': 1,
            'n_copies': 1,
            'open_clip_embedding_config': {
                'target': 'dragnuwa.svd.modules.encoders.modules.FrozenOpenCLIPImageEmbedder',
                'params': {
                    'freeze':True,
                }
            }
         }
        },
        {'input_key': 'fps_id',                 # vector
         'is_trainable': False,
         'ucg_rate': 0.1,
         'target': 'dragnuwa.svd.modules.encoders.modules.ConcatTimestepEmbedderND',
         'params': {
            'outdim': 256,
         }
        },
        {'input_key': 'motion_bucket_id',       # vector
        'ucg_rate': 0.1,
         'is_trainable': False,
         'target': 'dragnuwa.svd.modules.encoders.modules.ConcatTimestepEmbedderND',
         'params': {
            'outdim': 256,
         }
        },
        {'input_key': 'cond_frames',            # concat
         'is_trainable': False,
         'ucg_rate': 0.1,
         'target': 'dragnuwa.svd.modules.encoders.modules.VideoPredictionEmbedderWithEncoder',
        'params': {
            'en_and_decode_n_samples_a_time': 1,
            'disable_encoder_autocast': True,
            'n_cond_frames': 1,
            'n_copies': 1,
            'is_ae': True,
            'encoder_config': {
                'target': 'dragnuwa.svd.models.autoencoder.AutoencoderKLModeOnly',
                'params': {
                    'embed_dim': 4,
                    'monitor': 'val/rec_loss',
                    'ddconfig': {
                        'attn_type': 'vanilla-xformers',
                        'double_z': True,
                        'z_channels': 4,
                        'resolution': 256,
                        'in_channels': 3,
                        'out_ch': 3,
                        'ch': 128,
                        'ch_mult': [1, 2, 4, 4],
                        'num_res_blocks': 2,
                        'attn_resolutions': [],
                        'dropout': 0.0,
                    },
                    'lossconfig': {
                        'target': 'torch.nn.Identity',
                    }
                }
            }
        }
        },
        {'input_key': 'cond_aug',               # vector
         'ucg_rate': 0.1,
         'is_trainable': False,
         'target': 'dragnuwa.svd.modules.encoders.modules.ConcatTimestepEmbedderND',
        'params': {
            'outdim': 256,
        }
        }
    ]

    first_stage_config = {
        'loss_config': {'target': 'torch.nn.Identity'},
        'regularizer_config': {'target': 'dragnuwa.svd.modules.autoencoding.regularizers.DiagonalGaussianRegularizer'},
        'encoder_config':{'target': 'dragnuwa.svd.modules.diffusionmodules.model.Encoder',
        'params': { 'attn_type':'vanilla',
                    'double_z': True,
                    'z_channels': 4,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 2, 4, 4],
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0,
                }
            },
        'decoder_config':{'target': 'dragnuwa.svd.modules.autoencoding.temporal_ae.VideoDecoder',
                          'params': {'attn_type': 'vanilla',
                                     'double_z': True,
                                     'z_channels': 4,
                                     'resolution': 256,
                                     'in_channels': 3,
                                     'out_ch': 3,
                                     'ch': 128,
                                     'ch_mult': [1, 2, 4, 4],
                                     'num_res_blocks': 2,
                                     'attn_resolutions': [],
                                     'dropout': 0.0,
                                     'video_kernel_size': [3, 1, 1],
                }
            },
    }

    sampler_config = {
        'discretization_config': {'target': 'dragnuwa.svd.modules.diffusionmodules.discretizer.EDMDiscretization',
                                  'params': {'sigma_max': 700.0,},
        },
        'guider_config': {'target': 'dragnuwa.svd.modules.diffusionmodules.guiders.LinearPredictionGuider',
                          'params': {'max_scale':2.5,
                                     'min_scale':1.0,
                                     'num_frames':14},
        },
        'num_steps': 25,
    }

    scale_factor = 0.18215
    num_frames = 14

    ### others 
    seed = 42
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


args = Args()

def quick_freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.device = 'cpu'
        ### unet
        model = VideoUNet_flow(**args.network_config)
        self.model = OpenAIWrapper(model)

        ### denoiser and sampler
        self.denoiser = Denoiser(**args.denoiser_config)
        self.sampler = EulerEDMSampler(**args.sampler_config)

        ### conditioner
        self.conditioner = GeneralConditioner(args.conditioner_emb_models)

        ### first stage model
        self.first_stage_model = AutoencodingEngine(**args.first_stage_config).eval()

        self.scale_factor = args.scale_factor
        self.en_and_decode_n_samples_a_time = 1 # decode 1 frame each time to save GPU memory
        self.num_frames = args.num_frames   
        self.guassian_filter = quick_freeze(get_gaussian_kernel(kernel_size=args.kernel_size, sigma=args.sigma, channels=2))

        unet_lora_params, unet_negation = inject_lora(
                True, self, ['OpenAIWrapper'], is_extended=False, r=args.unet_lora_rank
            )
    
    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.sampler.device = self.device
        for embedder in self.conditioner.embedders:
            if hasattr(embedder, "device"):
                embedder.device = self.device
        return model_converted

    def train(self, *args):
        super().train(*args)
        self.conditioner.eval()
        self.first_stage_model.eval()
    
    def apply_gaussian_filter_on_drag(self, drag):
        b, l, h, w, c = drag.shape
        drag = rearrange(drag, 'b l h w c -> (b l) c h w')
        drag = self.guassian_filter(drag)
        drag = rearrange(drag, '(b l) c h w -> b l h w c', b=b) 
        return drag 

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = self.en_and_decode_n_samples_a_time # 1
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        for n in range(n_rounds):
            kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
            out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
            all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out