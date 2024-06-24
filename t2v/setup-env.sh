# create a virtual env
#conda create -n opensora python=3.10

# install torch
# the command below is for CUDA 12.1, choose install commands from 
# https://pytorch.org/get-started/locally/ based on your own CUDA version
#pip3 install torch torchvision

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
#pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
#git clone https://github.com/hpcaitech/Open-Sora
#cd Open-Sora
pip install -v .
