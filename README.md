# Free2CAD: Parsing Freehand Drawings into CAD Commands
![](docs/teaser.png)

## Introduction
This repository contains the implementation of [Free2CAD](http://geometry.cs.ucl.ac.uk/projects/2022/free2cad/) proposed in our SIGGRAPH 2022 paper.
* **Free2CAD: Parsing Freehand Drawings into CAD Commands**<br/>
By [Changjian Li](https://enigma-li.github.io/), [Hao Pan](http://haopan.github.io/), [Adrien Bousseau](http://www-sop.inria.fr/members/Adrien.Bousseau/), [Niloy Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/),<br/>
*ACM Transaction on Graphics (TOG), 41(4), 2022.*

It contains two parts: 1) **network training**, 2) **training dataset** and **trained network deployment** (e.g., for interactive modeling).

The code is released under the **MIT license**.

### Network training
This part contains the **Python** code for building, training and testing the nueral network using [TensorFlow](https://www.tensorflow.org/). 

💡 Great news: we have released the **docker file** used for training to ease the burden for configuration, please read README file within the *networkTraining* folder for more details.

### Training dataset and network deployment
This part contains the code for deploying the trained network in a C++ project that can be an interactive 3D modeling application. It also provides instructions to download the training dataset we generated, and our trained networks. 

Please read the README file in *dataAndModel* folder for more details.


## Citation
If you use our code or model, please cite our paper:

	@Article{Li:2022:Free2CAD, 
		Title = {Free2CAD: Parsing Freehand Drawings into CAD Commands}, 
		Author = {Changjian Li and Hao Pan and Adrien Bousseau and Niloy J. Mitra}, 
		Journal = {ACM Trans. Graph. (Proceedings of SIGGRAPH 2022)}, 
		Year = {2022}, 
		Number = {4}, 
		Volume = {41},
		Pages={93:1--93:16},
		numpages = {16},
		DOI={https://doi.org/10.1145/3528223.3530133},
		Publisher = {ACM} 
	}

 
 
## Contact
Any question you could contact Changjian Li (chjili2011@gmail.com) or Hao Pan (haopan@microsoft.com) for help.

