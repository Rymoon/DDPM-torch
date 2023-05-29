# DDPM-torch



**Install dependecies**
See reauirements.yml;

**register current package**
````bash
cd `path_to_repo_folder`
pip install -e .
````

**Replace path_to_images_folder**
In `ddpm.py::RUN_1`:

````python
dm = ImageDataModule([
Path(pkg_root,"../Datasets/CelebAHQ/data256x256").as_posix(),
Path(pkg_root,"../Datasets/CelebAHQ/data256x256_valid").as_posix(),
],image_size,batch_size)
````

If `CUDA_OUT_OF_RAM`,
* reduce batch_size;
* Or, reduce image_size;


If found `loss=nan`,
* reduce learning rate;
* Or, set `mseloss_reduction` to  "mean";

The available gpuid here can be limited by  
````python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = 1
# Before any torch-related import.
````
now ONLY `gpuid=1` is available.


### Reference
Codes modified from:
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master
- https://github.com/bojone/Keras-DDPM

