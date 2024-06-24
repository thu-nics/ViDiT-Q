## Setup:
### Installation:
Execute the following commands to setup Conda environment:
```
conda env create -f environment.yml
conda activate pixart
```
_Note: The environment provided in `environment.yml` is compatible with cu121. For other CUDA versions please install the correct versions of `torch`, `torchvision`, and `torchaudio`_

### Downloading model weights:
Download the corresponding model weights at the following links. For PixArt-alpha, please place the folders for the tokenizer and VAE weights under the same directory.

Model weights: \[[PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth)\, 
                 [PixArt-sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth)] <br>
Tokenizer and vae weights: \[PixArt-alpha: ([t5](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl),[vae](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema)), [PixArt-sigma](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers)\] 

## Running the quantization and quantized inference
To run the full model quantization and inference process, execute the command below with the following arguments:
```
VERSION = ... # model type (alpha or sigma)
SD_VAE = ... # path to text and image encoder weights
MODEL_PATH = ... # path to PixArt weights
PTQ_CONFIG = ... # quantization bit widths [w8a8, w4a8]
```
```
bash main.sh $VERSION $SD_VAE $MODEL_PATH $PTQ_CONFIG 
```
After executing the script above, generated images are saved in the directory `./quant_eval/{$PTQ_CONFIG}`

It is also possible to run only a part of the quantization/inference process by individually executing the corresponding command:

#### Step 1: Obtaining the calibration dataset:
```
python ./scripts/get_calib_data.py \
        --version "$VERSION" \
        --pipeline_load_from "$SD_VAE" \
        --model_path "$MODEL_PATH"
```
#### Step 2: Post-training quantization:
```
python ./scripts/ptq.py \
        --pipeline_load_from "$SD_VAE" \
        --model_path "$MODEL_PATH" \
        --ptq_config "$PTQ_CONFIG"
```
#### Step 3: Quantized inference:
```
python ./scripts/quant_txt2img.py \
        --version "$VERSION" \
        --pipeline_load_from "$SD_VAE" \
        --model_path "$MODEL_PATH" \
        --ptq_config "$PTQ_CONFIG" \
        --quant_act True \
        --quant_weight True
```
## Running inference with full precision:
To run inference with full precision, execute the following command:
```
python ./scripts/inference.py \
        --version "$VERSION" \
        --pipeline_load_from "$SD_VAE" \
        --model_path "$MODEL_PATH"
```