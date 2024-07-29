# [CVPR2024] Generate Like Experts: Multi-Stage Font Generation by Incorporating Font Transfer Process into Diffusion Models
This is the official Pytorch implementation of ***Generate Like Experts: Multi-Stage Font Generation by Incorporating Font Transfer Process into Diffusion Models***. 

## <center> Generate Like Experts: Multi-Stage Font Generation by Incorporating Font Transfer Process into Diffusion Models </center>

<center>Bin Fu, Fanghua Yu, Anran Liu, Zixuan Wang, Jie Wen, Junjun He, and Yu Qiao</center>

\
Few-shot font generation (FFG) produces stylized font images with a limited number of reference samples, which can significantly reduce labor costs in manual font designs. Most existing FFG methods follow the style-content disentanglement paradigm and employ the Generative Adversarial Network (GAN) to generate target fonts by combining the decoupled content and style representations. The complicated structure and detailed style are simultaneously generated in those methods, which may be the sub-optimal solutions for FFG task. Inspired by most manual font design processes of expert designers, in this paper, we model font generation as a multi-stage generative process. Specifically, as the injected noise and the data distribution in diffusion models can be well-separated into different sub-spaces, we are able to incorporate the font transfer process into these models. Based on this observation, we generalize diffusion methods to model font generative process by separating the reverse diffusion process into three stages with different functions: The structure construction stage first generates the structure information for the target character based on the source image, and the font transfer stage subsequently transforms the source font to the target font. Finally, the font refinement stage enhances the appearances and local details of the target font images. Based on the above multistage generative process, we construct our font generation framework, named MSD-Font, with a dual-network approach to generate font images. The superior performance demonstrates the effectiveness of our model.

* * *

## 1. TODO List
- [x] Add stage 1 training script. 
- [x] Add stage 2 training script. 
- [x] Add inference (sampling) script. 

## 2. Prerequisites

Our model is based on the LDM and Stable Diffusion platform, and you can use the following steps to install and compile our model. 

```
#################### install LDM ##########################
git clone https://github.com/CompVis/latent-diffusion.git   
mv ./latent-diffusion ./MSDFont      # change the folder's name
conda create -n MSDFont python=3.8.5
conda activate MSDFont
pip3 install torch torchvision torchaudio
pip install numpy
pip install albumentations
pip install opencv-python
pip install pudb
pip install imageio
pip install imageio-ffmpeg
pip install pytorch-lightning==1.6.5    # make sure the version of pytorch-lightning is 1.6.5
pip install omegaconf
pip install test-tube
pip install streamlit
pip install einops
pip install torch-fidelity
pip install transformers
cd MSDFont
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
#################### update LDM to stable diffusion ##########################
cd ..
git clone https://github.com/Stability-AI/stablediffusion.git  
cp -fr ./stablediffusion/* ./MSDFont/ # copy and cover the files
cd MSDFont
pip install diffusers invisible-watermark
pip install -e .
#conda install xformers -c xformers      # option, more efficiency and speed on GPUs
#################### MSD-Font ##########################
cd .. 
git clone https://github.com/fubinfb/MSD-Font.git
cp -fr ./MSD-Font/* ./MSDFont/ # copy and cover the files
cd MSDFont
pip install fonttools
pip install Pillow==9.5.0   # please make sure the version of Pillow <= 9.5.0
```

## 3. Datasets

In this project, we select the native true-type font (TTF) formats for datasets, which is storage-efficient and easy-to-use. The structure of the dataset is constructed following the instructions from [FFG-benchmarks](https://github.com/clovaai/fewshot-font-generation). 

You can collect your own fonts from the following web site (for non-commercial purpose):

- [https://www.foundertype.com/index.php/FindFont/index](https://www.foundertype.com/index.php/FindFont/index) (acknowledgement: [DG-Font](https://github.com/ecnuycxie/DG-Font) and [FFG-benchmarks](https://github.com/clovaai/fewshot-font-generation) refer this web site)

We also provide an example of dataset structure in the folder "fontdata_example". 


## 4. Training
Different from GAN-based Font Generation Methods, the diffusion based model needs more training steps to converge, since the diffusion model has additional dimention "time (diffusion step)" need to be optimized. In our model, we keep training our model and save the model every two epochs. We recommend to optimize the model at least 80 epoches for better converged. 

In our model, we use VQ-VAE encoder to project font images into latent space. The parameters of VQ-VAE are provided in [v2-1_512-ema-pruned.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt). 

### 4.1. Stage 1-1: Training the $E_c^1$, $E_s^1$, and $\tilde{z}_{\theta_1}^{(g,1)}(\tilde{z}_t,t,y_1)$ for Font Transer Stage

##### Modify the configuration file

The configuration file: configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml

Please read and modify the configuration file: 
```
edit_t1: the value of t1
edit_t2: the value of t2
ckpt_path: the path of the file: v2-1_512-ema-pruned.ckpt
data_dirs: the path of the training set
train_chars: the path of the json file for training characters
source_path: the path of the ttf file for source font
```
and you can also modify other settings in this file. 

##### Run training script
```
CUDA_VISIBLE_DEVICES=GPUID python main.py --base configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml -t --gpus 0,
```


### 4.2. Stage 1-2: Training the $E_c^2$, $E_s^2$, and $\tilde{z}_{\theta_2}^{(g,2)}(\tilde{z}_t,t,y_2)$ for Font Refinement Stage


##### Modify the configuration file

The configuration file: configs/MSDFont/MSDFont_Train_Stage1_rec_model_predx0_miniUnet.yaml

Please read and modify the configuration file: 
```
ckpt_path: the path of the file: v2-1_512-ema-pruned.ckpt
data_dirs: the path of the training set
train_chars: the path of the json file for training characters
source_path: the path of the ttf file for source font
```
and you can also modify other settings in this file. 

##### Run training script
```
CUDA_VISIBLE_DEVICES=GPUID python main.py --base configs/MSDFont/MSDFont_Train_Stage1_rec_model_predx0_miniUnet.yaml -t --gpus 0,
```

### 4.3. Stage 2: Finetune $E_c^2$, $E_s^2$, and $\tilde{z}_{\theta_2}^{(g,2)}(\tilde{z}_t,t,y_2)$ for Font Refinement Stage

After finish the stage 1 training, we use $E_c^1$, $E_s^1$, and $\tilde{z}_{\theta_1}^{(g,1)}(\tilde{z}_t,t,y_1)$ to finture $E_c^2$, $E_s^2$, and $\tilde{z}_{\theta_2}^{(g,2)}(\tilde{z}_t,t,y_2)$ in stage 2. To keep the batch size same with stage 1, we use model parallel in this stage by putting above two set of models into two different GPUs. 

As pytorch-lightning not soppurt this opperation, we need:

(1). add the script"./add_files/ddp_distri.py" to your conda envs: in the file "~/anaconda3/envs/MSDFont/lib/python3.8/site-packages/pytorch_lightning/strategies/", 

(2). modify the "~/anaconda3/envs/MSDFont/lib/python3.8/site-packages/pytorch_lightning/strategies/\_\_init\_\_.py" from:

```
from pytorch_lightning.strategies.bagua import BaguaStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp import DDPStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp2 import DDP2Strategy  # noqa: F401
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from pytorch_lightning.strategies.dp import DataParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.horovod import HorovodStrategy  # noqa: F401
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.ipu import IPUStrategy  # noqa: F401
from pytorch_lightning.strategies.parallel import ParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded import DDPShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from pytorch_lightning.strategies.single_hpu import SingleHPUStrategy  # noqa: F401
from pytorch_lightning.strategies.single_tpu import SingleTPUStrategy  # noqa: F401
from pytorch_lightning.strategies.strategy import Strategy  # noqa: F401
from pytorch_lightning.strategies.strategy_registry import call_register_strategies, StrategyRegistry  # noqa: F401
from pytorch_lightning.strategies.tpu_spawn import TPUSpawnStrategy  # noqa: F401

STRATEGIES_BASE_MODULE = "pytorch_lightning.strategies"

call_register_strategies(STRATEGIES_BASE_MODULE)
```
to
```
from pytorch_lightning.strategies.bagua import BaguaStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp import DDPStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp2 import DDP2Strategy  # noqa: F401
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from pytorch_lightning.strategies.dp import DataParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.horovod import HorovodStrategy  # noqa: F401
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.ipu import IPUStrategy  # noqa: F401
from pytorch_lightning.strategies.parallel import ParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded import DDPShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from pytorch_lightning.strategies.single_hpu import SingleHPUStrategy  # noqa: F401
from pytorch_lightning.strategies.single_tpu import SingleTPUStrategy  # noqa: F401
from pytorch_lightning.strategies.strategy import Strategy  # noqa: F401
from pytorch_lightning.strategies.strategy_registry import call_register_strategies, StrategyRegistry  # noqa: F401
from pytorch_lightning.strategies.tpu_spawn import TPUSpawnStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp_distri import DDP_Distri_Strategy  #### add this line for model parallel

STRATEGIES_BASE_MODULE = "pytorch_lightning.strategies"

call_register_strategies(STRATEGIES_BASE_MODULE)
```

##### Modify the configuration file

The configuration file: configs/MSDFont/MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri.yaml

Please read and modify the configuration file: 
```
edit_t1: the value of t1, keep the same with Stage 1-1 
edit_t2: the value of t2, keep the same with Stage 1-1 
ckpt_path: the path of the saved model in Stage 1-2, eg: logs/rec_stage_epoch=79-step=799999.ckpt
data_dirs: the path of the training set
train_chars: the path of the json file for training characters
source_path: the path of the ttf file for source font

trans_model_config:
      config_path: the config for Stage 1-1, eg: "configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml"
      model_path: the path of the saved model in Stage 1-1, eg: "logs/trans_stage_epoch=79-step=799999.ckpt"
```
and you can also modify other settings in this file. 

##### Run training script
```
CUDA_VISIBLE_DEVICES=GPUID1,GPUID2 python main_distri.py --base configs/MSDFont/MSDFont_Train_Stage2_rec_model_predx0_miniUnet_distri.yaml -t --gpus 0,
```


## 5. Inference

##### Modify the configuration file

The configuration file: infer_MSDFont.sh

Please read and modify the configuration file: 
```
--outdir output file, eg: results/infer_result \
--path_genchar target characters, eg: FontData/chn/eval_unseen_chars.json \
--path_refchar reference characters, eg: FontData/chn/ref_chars.json \
--path_ttf target font, eg: FontData/chn/ttfs/infer_unseen_font \
--source_path source font, eg: FontData/chn/source.ttf \
--path_config_rec config of rec model, provided in: configs/MSDFont/MSDFont_Eval_rec_model_predx0_miniUnet.yaml \
--path_rec_model the path of the saved model in Stage 2, eg: logs/rec_stage2_epoch=79-step=799999.ckpt \
--path_config_trans config of trans model, provided in: configs/MSDFont/MSDFont_Eval_trans_model_predx0_miniUnet.yaml \
--path_trans_model the path of the saved model in Stage 1-1, eg: logs/trans_stage_epoch=79-step=799999.ckpt
```
and you can also modify other settings in this file. 

##### Run script
```
sh infer_MSDFont.sh
```



## Code license

This project is distributed under [MIT license](LICENSE).

## Acknowledgement

This project is based on [FFG-benchmarks](https://github.com/clovaai/fewshot-font-generation), [LDM](https://github.com/CompVis/latent-diffusion), and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).

## How to cite

```
@InProceedings{Fu_2024_CVPR,
    author    = {Fu, Bin and Yu, Fanghua and Liu, Anran and Wang, Zixuan and Wen, Jie and He, Junjun and Qiao, Yu},
    title     = {Generate Like Experts: Multi-Stage Font Generation by Incorporating Font Transfer Process into Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {6892-6901}
}
```
