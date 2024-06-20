# This code is the source code implementation for the paper "Personalized federated distillation Differential Privacy based on Parameter decoupling"



## Abstract

![](/pic/arc.png)

Personalized Federated Learning (PFL), as a novel Federated Learning (FL) paradigm, is capable of generating personalized models for heterogenous clients. However, currently popular Personalized Federated Learning often faces the phenomenon of weight divergence and the risk of privacy leakage. In this paper, we propose DP-pFedCKD, it consists of two main parts: pFedCKD and ACDP. pFedCKD uses knowledge distillation techniques to impose consistency regularization between blocks to mitigate different types of data heterogeneity. In addition, the sensitivity of the uploaded model parameters is determined by the clipping threshold. The clipping threshold is closely related to the magnitude of the clipping model parameter norm and the amount of added differential privacy noise. Therefore, we propose a clipping threshold dynamic decay strategy ACDP to balance the bias due to clipping and the error introduced by Gaussian noise. Finally, we give an experimental analysis and privacy analysis on five real datasets, which proves the effectiveness of the above method.



## Experimental Environment

**Installation：**

To run the code, you need to install the following packages：

```
cuda                     11.6.1                      
dp-accounting            0.4.2                    
h5py                     3.8.0                    
keras                    2.12.0                   
matplotlib               3.7.1                    
numpy                    1.23.1                   
oauthlib                 3.2.2                  
opacus                   1.3.0                                                  
python                   3.9.16                      
scikit-learn             0.23.1                   
scipy                    1.10.1                
six                      1.16.0                        
torch                    1.13.1                   
torchaudio               0.13.1                   
torchsummary             1.5.1                    
torchvision              0.14.1                                     
```

## Datasets

```
CIFAR-10
CIFAR-100
FMNIST
EMNISYT
SVHN
```

## Experimental Setup

**Hyperparameters:**

- Training is conducted over 100 rounds with 10 clients participating. Each client executes 4 epochs per round.
- The local learning rate is set at 0.01, and the batch size is 128.
- For CIFAR-10, clients' local data is limited to 4 classes to simulate heterogeneous settings. For CIFAR-100, clients' data is limited to 40 classes.
- The DP-pFedCKD method uses a dynamic threshold decay strategy for clipping and Gaussian noise addition for differential privacy.

**Models:**

- **Simple CNN**: Used for FMNIST and EMNIST datasets, consisting of convolutional layers followed by fully connected layers.
- **ResNet-18**: Used for SVHN, CIFAR-10, and CIFAR-100 datasets, with 18 convolutional layers organized into 4 blocks, followed by a global average pooling layer and a fully connected layer.

**Privacy-Preserving Methods:**

- **Differential Privacy (DP)**: Gaussian noise is added to the model updates to ensure user-level differential privacy.
- **Adaptive Clipping Threshold**: A dynamic decay strategy is used to adjust the clipping threshold during training, balancing the trade-off between privacy and model utility.

**Evaluation Metrics:**

- **Model Performance**: The accuracy of the global and personalized models is used as the primary metric.
- **Privacy Protection**: Ensured by satisfying differential privacy requirements through the addition of noise to model updates.
- **Communication Cost**: Measured by the amount of data uploaded during the training process.



## Experimental Results

The experiments in this paper validate the effectiveness of the proposed DP-pFedCKD framework through various tests conducted on five real-world datasets: FMNIST, EMNIST Balanced, SVHN, CIFAR10, and CIFAR100. The experiments are designed to assess the model's performance under different non-IID data settings, including pathological and Dirichlet distributions. Key evaluation metrics include model accuracy and communication cost. The results demonstrate that DP-pFedCKD consistently outperforms baseline methods in terms of accuracy and privacy preservation, especially in heterogeneous data environments. The dynamic threshold decay strategy (ACDP) effectively balances the trade-off between privacy and utility, ensuring robust performance across different datasets.

![](/pic/1.png)

![](/pic/2.png)

![](/pic/3.png)

![](/pic/4.png)



