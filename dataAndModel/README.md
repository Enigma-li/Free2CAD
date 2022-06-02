# Training dataset and network deployment

In this part, there will be instructions for downloading data, checkpoint and sample code for deploy network in C++ applications.

## Training data

Details about data rendering, please refer to the paper in our [project page](http://geometry.cs.ucl.ac.uk/projects/2022/free2cad/). Totally, we generate about 160k data items for training and 50k data items for testing (with data augmentation), with size of about **26.5GB** and **5.93GB** after two passes of compression (zlib and TFRecord).

Now we provide the link for downloading training datasets:

>[Training data](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/EnPDEYSkfUBNshmvXP_s9MsBFobNIgmgBdhm44m7FYp8qg?e=jUZxhc)


## Trained networks

We provide links for downloading the checkpoints of our trained networks:
>[Checkpoint](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/EnngouKiMzZOvUceg-HdsjUBA1KZVmQX1wN7Z629ZwKFxw?e=lwoTEm)

## Network deployment

To deploy the trained network in C++ project in Windows, users must compile and build the TensorFlow libs and dlls from source using the ***SAME*** version as in network training stage. Then the source code named `trained_network.h` and `trained_network.cpp` provide a way to use the network in C++. We also provide the pseudo code `demo_pseudo.cpp` to use the trained networks. 

💡💡💡 ***Tips***:
* We provide the detailed input/output channels, sizes, etc., in the trained_model.cpp file, you can find the meanings easily.
* The first network forward pass would be time-consuming (about 4s on 2080Ti GPU) because of the initialization of GPU and CUDA settings. So after loading the network, please first execute the `warmup` step, all other forward passes after this `warmup` would be fast, i.e., 40ms.
* Compiling and building TensorFlow from source under Windows is time consuming (*over 1 hours*), we use **Visual Studio 2019** to build **TensorFlow 2.3** by refering to this great [blog](https://medium.com/vitrox-publication/deep-learning-frameworks-tensorflow-build-from-source-on-windows-python-c-cpu-gpu-d3aa4d0772d8), which works for us, other configurations are **not tested**. 
* Download the pre-built Tensorflow 2.3 [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/changjli_connect_hku_hk/EtpvphFoR81NnBBv567HyBkBubWnus8bovpR4Z2oUX_0WA?e=5IugoA).

