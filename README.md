#TCMCL

This is the open-source code for the paper "Text-Centric Multimodal Contrastive Learning for Sentiment Analysis"".

To reproduce this work, the following steps are mainly required:

1. Environment Preparation
   
No special requirements, this work uses Python 3.7 and Torch 1.9.0.

Refer to requirement.txt for detailed environment configuration.

2. Dataset Download and Configuration

Inside the `./datasets` folder, run `./download_datasets.sh` to download the MOSI and MOSEI datasets.

The `configs.py` file is used to configure the parameters used for the two datasets.

3. Running

```shell
python main.py --dataset [Specify which dataset to use]
```
