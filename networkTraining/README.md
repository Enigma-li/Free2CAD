# Network Training

We train our network on a server running Linux system, the training script is **ONLY** tested on Linux system.

Clone the repository and enter the network training part:

    git clone https://github.com/Enigma-li/Free2CAD.git
    cd networkTraining

There are three sub-folders under the training project root ***networkTraining*** folder.
* *libs* folder contains the custom training data decoder implemented in C++ and imported as custom ops in TensorFLow framework.
* *script* folder contains the network building, data loading, training and testing scripts.
* *utils* folder contains the utility functions.


##  Installation

We highly recommend to use docker to maintain the developing environment, and we have released the docker image we used for this project to help ease the configuration burden, so you can either download and load the image or you can configure the environment from scrath.

### Free2CAD Docker Image

You can download the docker image from the link：[Free2CAD_DockerImage](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/EgHtViQo535IszMODFxn1HgB8hbRD66dEzEyOsdiiLhCyQ?e=qU7MuG), after downloading, you can simply load it, then you are ready to train/test the nueral networks.

### Configure from scratch

Below are the main package/library versions that are used:

* TensorFlow: ***2.3.0***
* Python: ***3.6***
* Cuda: ***10.1***
* Cudnn: ***7.6***

Other packages could be installed via `pip` if needed.

### Install

We first build the custom ops in *libs* folder and then configure the Linux system to compatible with the training script. You will see the file named `custom_dataDecoder.so` generated in `libs` folder after building the ops.

* Enter *libs* folder and build the ops. Remember to change the TensorFlow source path in `build.sh` based on your system configuration.
  > cd libs <br /> ./build.sh

## Usage
With the project installed and custom ops built successfully, now you could try the training and testing script.

Enter the `script` folder, you will see some files whose names are beginning with "train", these will be the training and testing scripts. We accept the console parameters for training and testing configuration, just type python with `-h/--help` command to see the usage, e.g., 

    $python train_Grouper.py -h

Then you will get:

    usage: train_Regressor.py [-h] --dbDir DBDIR --outDir OUTDIR --embedS_ckpt
                              EMBEDS_CKPT --gpTF_ckpt GPTF_CKPT --devices DEVICES
                              [--ckpt CKPT] [--cnt CNT] [--status STATUS]
                              [--bSize BSIZE] [--maxS MAXS]

    optional arguments:
        -h, --help            show this help message and exit
        --dbDir DBDIR         TFRecords dataset directory
        --outDir OUTDIR       output directory
        --embedS_ckpt EMBEDS_CKPT
                              stroke embedding checkpoint
        --gpTF_ckpt GPTF_CKPT 
                              grouping transformer checkpoint
        --devices DEVICES     GPU device indices
        --ckpt CKPT           checkpoint path
        --cnt CNT             continue training flag
        --status STATUS       training or testing flag
        --bSize BSIZE         batch size
        --maxS MAXS           maxinum sliding window width

💡Note:
* You can specify 'train' or 'test' for --status, then you can train or test the neural network
* DEVICES specifies which GPU is used, we now only support single GPU training
* Parameters, e.g., dbDir, outDir, checkpoint, should be specified by users
* *Other training parameters are hardcoded at the very beginning of each script (**hyper_params**), you could change them to some values you want and it is easy to get the meaning of them*
* Use tensorboard to check the training status, there are curves and figures


### Traing From Scratch

We split the traing into three stages: stroke embedding, grouper, and regressor training, both grouper and regressor training is depend on the stroke embedding, so it should be trained first. Then, you should train the grouper and regressor in order, i.e.:

    # 1. Embedding
    $python train_AES_SG.py --dbDir=/data/path --outDir=/output/path --codeSize=256 --devices='0' --statsus='train'

    # 2. Grouper
    $python train_Grouper.py --dbDir=/data/path --outDir=/output/path --embed_ckpt=/path/to/embedding/checkpoint --devices='0' --status='train' --d_model=256

    # 3. Regressor
    $python train_Regressor.py --dbDir=/data/path --outDir=/output/path --embed_ckpt=/path/to/embedding/checkpoint --gpTF_ckpt=/path/to/grouper/checkpoint --devices='0' --status='train' 

💡Note:
* After training each network, the checkpoint is written with two files: ckpt-x.data-00000-of-00001 and ckpt-x.index, we will load them into C++ project using Tensorflow C++ API
* to use the trained network in C++ project in Windows, you should compile and build the ***SAME*** TensorFlow version, see more details in *Data generation and network deployment* part.
