"""

Refactor based on MyDDPM.apps.ddpm2.py


Concepts:
- state: float tensor, range[-1,1], shape of CHW;
- batch_state: a bathc of state, shape of BCHW;
- image: float tensor, range[0,1], shape of HW3
- raw: any data load from disk.
- sample: return a state;
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import skimage
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from myddpm_pytorch.apps.rennet import (call_by_inspect, getitems_as_dict,
                                        root_Results)


# ==== Utils ====
def list_pictures(directory, ext:Tuple|str=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif',
                                  'tiff')):
    """Lists all pictures in a directory, including all subdirectories.

    # Arguments
        directory: string, absolute path to the directory
        ext: tuple of strings or single string, extensions of the pictures

    # Returns
        a list of paths


    # Copy from keras_preprocessing.image.utils::list_pictures
    """
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]




def batch_tensor_save(p,t:torch.Tensor):
    """cat as a grid

    Assume tensor in float,range[0,1], BCHW.
    """
    imgs = [t_ for t_ in t.detach().cpu()] # List of CHW
    imgs = [t_.permute(1,2,0).numpy() for t_ in imgs]
    n = len(imgs)
    h,w,c = imgs[0].shape
    grid_size = math.ceil(math.sqrt(n))
    figure = np.zeros((h*grid_size,w*grid_size,c))
    for i in range(grid_size):
        for j in range(grid_size):
            img_ = imgs[i*grid_size+j]
            figure[i*h:(i+1)*h, j*w: (j+1)*w,:] = img_
    
    plt.imsave(p,figure)
    return figure, grid_size

def batch_state_save(p,state:torch.Tensor):
    """clip and save as image grid

    Assume state: tensor float,range[-1,1], BCHW.
    """
    b_img=  (state+1)/2
    b_img = torch.clip(b_img,0,1)
    return batch_tensor_save(p,b_img)
# ===============

from typing import List

import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """

    """
    def __init__(self, folder:str|List[str], image_size: int):
        super().__init__()

        # List of files
        # self._files = [p for p in Path(folder).glob(f'**/*.jpg')]
        self._files = list_pictures(folder) if isinstance(folder,str) else sum((list_pictures(p) for p in folder),[])
        

        # Transformations to resize the image and convert to tensor
        # Remap values to [-1,1],
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)
    

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, folder:str|List[str], image_size: int,batch_size, num_workers = 16):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.folder = folder
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Initialize your dataset here
        self.dataset = ImageDataset(self.folder, self.image_size)

    def train_dataloader(self):
        # Return a DataLoader for the training data
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True,num_workers=self.num_workers)

 

from myddpm_pytorch.apps.ddpm_m import p_sample


def sample(n_samples, eps_model , image_channels, image_size, n_steps, device, alpha_bar, alpha, sigma2):
        """
        ### Sample images

        Return Tensor(batch_size, C, H, W)

        and H=W
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([n_samples, image_channels, image_size, image_size],device=device)

            # Remove noise for $T$ steps
            for t_ in trange(n_steps, desc="sample"):
                # $t$
                t = n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = p_sample(x, x.new_full((n_samples,), t, dtype=torch.long),
                            eps_model=eps_model,
                            alpha_bar=alpha_bar,
                            alpha=alpha,
                            sigma2=sigma2)
        return x

from pytorch_lightning import LightningModule
class EMA(pl.Callback):
    def __init__(self,decay=0.999,optimizer_idx=0,*,apply_by="epoch",collect_by="epoch"):
        """
        (apply_by,collect_y):
        - ("epoch","epoch")
        - ("epoch","batch")

        Mantain a separete model, "ema", as average of original one; Never affect original optimization process; But used for compute metrics and predict or stop criterion;
        """
        self.decay = decay
        self.optimizer_idx = optimizer_idx
        assert (apply_by,collect_by) in [("epoch","epoch"),("epoch","batch")]
        self.apply_by = apply_by
        self.collect_by = collect_by
        self.shadow_params = {}
        self.backup_params = {}
    
    def ema_collect(self):
        for group in self.optimizer.param_groups:
            for name, param in group['params']:
                if param.requires_grad:
                    p = param.data
                    if id(param) in self.shadow_params:
                        sp = self.shadow_params[id(param)]
                        self.shadow_params[id(param)] -= (1.0 - self.decay) * (sp - p )
                    else:
                        self.shadow_params[id(param)]= p
    def ema_apply(self):
        for group in self.optimizer.param_groups:
            for name, param in group['params']:
                if param.requires_grad:
                    if id(param) in self.shadow_params:
                        self.backup_param[id(param)] = param.data
                        param.data = self.shadow_params[id(param)]
    def ema_restore(self):
        for group in self.optimizer.param_groups:
            for name, param in group['params']:
                if param.requires_grad:
                    if id(param) in self.shadow_params:
                        if id(param) in self.backup_params:
                            param.data = self.backup_params[id(param)]

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """(after backward and update of batch)
        collect
        """
        
        if self.collect_by == "batch":
            self.ema_collect()

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
 
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """restore
        """
        
        if self.apply_by in ("epoch",):
            self.ema_restore()
        
                        
        return super().on_train_epoch_start(trainer, pl_module)
    

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """
        collect, backup-and-apply
        """
        
        if self.collect_by == "epoch":
            self.ema_collect()
        
        if self.apply_by == "epoch":
            self.ema_apply()
                        
        return super().on_train_epoch_end(trainer, pl_module)
    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_shadow_params":self.shadow_params,
            "ema_backup_params":self.backup_params,
        }
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.shadow_params = state_dict["ema_shadow_params"]
        self.backup_params = state_dict["ema_backup_params"]
        return super().load_state_dict(state_dict)
    

class Model(LightningModule):
    """From /annotated_deep_learning_paper_implementations/labml_nn/diffusion/ddpm/experiment.py:: Config class

    Assume values/states in [-1,1].
    """
    def __init__(self,n_steps,device,*,
                 image_channels=3,
                 image_size = 128,
                 n_channels=64,
                 ch_mults=[1, 2, 2, 4],
                 is_attn=(False,False,True,True),
                 n_blocks=2,
                 learning_rate=2e-5,
                 mseloss_reduction="mean",
                 extra_info:dict={}):
        
        """
        * `n_steps`: $T$
        * `extra_info`: Only for save_hyperparameters
        """
        super().__init__()
        self.save_hyperparameters(ignore=["device"])
        self.image_channels = image_channels
        self.image_size = image_size
        self.n_steps = n_steps
        self.mseloss_reduction = mseloss_reduction
    

        from myddpm_pytorch.apps.ddpm_m import DenoiseDiffusion, UNet
        self.eps_model = UNet(
            # Number of channels in the image. $3$ for RGB.
            image_channels=image_channels,
            # Number of channels in the initial feature map
            # Time embedding has `n_channels * 4` channels
            n_channels=n_channels,
            # The list of channel numbers at each resolution.
            # The number of channels is `channel_multipliers[i] * n_channels`
            ch_mults=ch_mults,
            # The list of booleans that indicate whether to use attention at each resolution
            is_attn=is_attn,
            #  The number of `UpDownBlocks` at each resolution
            n_blocks=n_blocks)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=n_steps,
            device=device,
        )
        
        self.learning_rate= learning_rate

    def configure_optimizers(self):
        """
        Return [optimizers],[schedulers]
        Or {}
        """
        # optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        # return {"optimizer": optimizer, "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,min_lr=2e-6,factor=0.7,patience=5), "monitor": "train_loss"}
        # 
        optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1)
        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler":scheduler,
                "interval":"epoch",
                "frequency":2,
            }
        }


    
    def sample(self, n_samples):
        """
        ### Sample the state x

        Return (batch_size, H, W, C)
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in trange(self.n_steps, desc="sample"):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
        return x
    
    
    
    def training_step(self, batch, batch_idx):
        """
        Return training_loss
        """
        loss = self.diffusion.loss(batch,reduction=self.mseloss_reduction)
        self.log('train_loss', loss)
        return loss
    
        

class SampleEachEpoch(pl.Callback):
    def __init__(self,sub_folder="sample",n_samples=16, epoch_interval=1):
        super().__init__()
        self.sub_folder = sub_folder
        self.n_samples = n_samples
        assert epoch_interval>0 
        self.epoch_interval = epoch_interval

    def on_epoch_end(self,trainer:pl.Trainer,pl_module)->None:
        epoch = trainer.current_epoch # Assigned by pytorch_lightning
        if epoch %self.epoch_interval == 0 and epoch>0:
            x_samp = trainer.model.sample(self.n_samples)


            log_dir = trainer.log_dir
            p= Path(log_dir,self.sub_folder)
            if not p.exists():
                p.mkdir()
            pp= Path(p,f"epoch_{epoch:06}_SampleGrid.png")
            batch_state_save(pp.as_posix(),x_samp)
            print(f"- save image: {pp.as_posix()}\n")
        return super().on_epoch_end(trainer,pl_module)
    
import time
from datetime import timedelta
class ElapsedTime(pl.Callback):
    def __init__(self):
        self.tic = None
        self.toc = None
    def on_fit_start(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        self.tic = time.time()
        return super().on_fit_start(trainer, pl_module)
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        self.toc = time.time()
        print("- elapsed time: ",timedelta(seconds=self.toc-self.tic),"\n")
        return super().on_epoch_end(trainer, pl_module)
    def on_fit_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        self.toc = time.time()
        print("- elapsed time: ",timedelta(seconds=self.toc-self.tic),"\n")
        return super().on_fit_end(trainer, pl_module)

def RUN_sancheck():
    DEBUG = True
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    import myddpm_pytorch

    pkg_root = Path(myddpm_pytorch.__file__).parent
    root_dir=  Path(pkg_root,"../Results",Path(__file__).stem+"_sancheck").as_posix()
    gpuid = 0

    batch_size = 16
    image_size = 128
    T = 1000
    max_epochs = 3600*3
    learning_rate = 2e-4
    
    dm1 = ImageDataModule([
        Path(pkg_root,"../Datasets/CelebAHQ_1").as_posix(),
        ],image_size,batch_size)

    trainer = pl.Trainer(default_root_dir=root_dir,gpus=[gpuid],
                         val_check_interval=0.0,# Disable validation
                         max_epochs=max_epochs, # You can stop bt Ctrl+C once
                         callbacks=[
                             SampleEachEpoch(epoch_interval=600),
                             ElapsedTime(), # Show elapsed time from start of fit
                             #ModelCheckpoint()# Customized weights saving
                            ],
                         )
    log_dir = trainer.log_dir
    print("- log_dir:",str(log_dir))
    Path(log_dir,"sample").mkdir(parents=True,exist_ok=True)
    
    model = Model(T,torch.device(f"cuda:{gpuid}"),learning_rate=learning_rate)
    trainer.fit(model,datamodule=dm1)
    return locals()

def RUN_1():
    DEBUG = False
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    import myddpm_pytorch

    pkg_root = Path(myddpm_pytorch.__file__).parent
    root_dir=  Path(pkg_root,"../Results",Path(__file__).stem).as_posix()
    gpuid = 0

    batch_size = 16
    image_size = 128
    T = 1000
    max_epochs = 20*36
    learning_rate = 6e-5
    mseloss_reduction = "mean"

    dm = ImageDataModule([
        Path(pkg_root,"../Datasets/CelebAHQ/data256x256").as_posix(),
        Path(pkg_root,"../Datasets/CelebAHQ/data256x256_valid").as_posix(),
        ],image_size,batch_size)
    
    trainer = pl.Trainer(default_root_dir=root_dir,gpus=[gpuid],
                         val_check_interval=0.0,# Disable validation
                         max_epochs=max_epochs, # You can stop bt Ctrl+C once
                         callbacks=[
                             SampleEachEpoch(epoch_interval=20),
                             #ModelCheckpoint()# Customized weights saving
                            ]
                         )
    log_dir = trainer.log_dir
    print("- log_dir:",str(log_dir))
    Path(log_dir,"sample").mkdir(parents=True,exist_ok=True)
    
    
    model = Model(T,torch.device(f"cuda:{gpuid}"),learning_rate=learning_rate,mseloss_reduction=mseloss_reduction)
    trainer.fit(model,datamodule=dm)
    return locals()


if __name__ == '__main__':
    """
    {root_dir}/version_{}:
        /lightning_logs/ ...weights
        /sample/ ...training_samples

    Assume that:
    - 
    
    """
    RUN_1()