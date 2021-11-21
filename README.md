# deepManReg: a deep manifold-regularized learning model for improving phenotype prediction from multi-modal data

## Summary
The phenotypes of complex biological systems are fundamentally driven by various multi-scale mechanisms. Increasing multi-modal data enables a deeper understanding of underlying complex mechanisms across scales for the phenotypes such as single cell multi-omics data for cell types. However, integrating and interpreting such large-scale multi-modal data remains challenging, especially given highly heterogeneous, nonlinear relationships across modalities. To address this, we developed an interpretable regularized learning model, deepManReg to predict phenotypes from multi-modal data. First, deepManReg employs deep neural networks to learn cross-modal manifolds and then align multi-modal features onto a common latent space. This space aims to preserve both global consistency and local smoothness across modalities and to reveal higher-order nonlinear cross-modal relationships. Second, deepManReg uses cross-modal manifolds as a feature graph to regularize the classifiers for improving phenotype predictions and also prioritizing the multi-modal features and cross-modal interactions for the phenotypes. We applied deepManReg to (1) the image data of handwritten digits with multi-features and (2) recent single cell multi-modal data (Patch-seq data) including transcriptomics and electrophysiology for neuronal cells in the mouse brain. After comparing with the state-of-the-arts, we show that deepManReg has significantly improved predicting phenotypes in both datasets, and also prioritized genes and electrophysiological features for the phenotypes of neuronal cells 

## Flow chart
![alt text](https://github.com/daifengwanglab/deepManReg/blob/main/figures/workflow.png)

## Installation

This script need no installation, but has the following requirements:
* PyTorch 0.4.1 or above
* Python 3.6.5 or above

## Demo for aligning and classifying single-cell multi-modal data

### Sample set: 3654 cells in the mouse visual cortex

We provided a multi-modal data set from mouse visual cortex to test our method.

-  `/visual_sample/phase1.ipynb` calculates the deep manifold alignemnt feature latent space.
-  `/visual_sample/phase2.ipynb` compares the accuracy of DeepManReg with other methods in terms of classifying cell layers and cell t-types, by fitting a regularized neruon network.
-  `/visual_sample/boxplot_visual.r` draws the boxplot out of output accuracys in phase 2.
-  `/visual_sample/igraph_visual.r` draws the igraph for phase 1 feature latent space.

A demo for aligning and aligning single-cell multi-modal data is available at `/visual_sample/phase1.ipynb` and `/visual_sample/phase2.ipynb` for a Patch-seq dataset (3654 cells) in the mouse visual cortex (https://github.com/berenslab/layer4). The input data includes the gene expression levels and electrophysiological features of these cells in a rda file. Also, we provide a list of neuronal marker genes as select gene features. The code performs alignment on a reduced-dimensional space on single-cell electrophysiological data and gene expression with three major methods: deepManReg, Linear manifold alignment (NMA), Canonical correspondence analysis (CCA) and MATCHER.  The expected output includes the visualization of the latent space after each alignment method. The total running time of this demo was a couple of hours on a local laptop, which includes both alignment (Phase 1) and regularized classification (Phase 2). Phase 2 actually took most of the time and has no big difference across methods. The major difference of the running times between deepManReg and others happened in Phase 1, alignment: CCA (725.96 seconds), Manifold Alignment (663.43 seconds), MATCHER(150.94 seconds), and deepManReg with GPUs (57.90 seconds), with CPU (90.10 seconds).

Others plots can be generated through the following codes files: `igraph_visual.Rmd` visualizes the gene regulatory network by igraph for each cluster; `boxplot_visual.Rmd` visualizes the boxplot comparing classification results between deepManReg and other methods, e.g., CCA, Linear Manifold Alignment, MATCHER.

## Usage

Import libraries

```python
import numpy as np
import scipy.spatial.distance as sd
from deepManReg.neighborhood import neighbor_graph, laplacian
from deepManReg.correspondence import Correspondence
from deepManReg.stiefel import *
from deepManReg.deepManReg import *
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Phase 1: Deep Manifold Alignment

First, define the two neural networks which are the same as in this case:

```python
class Net(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h1_sigmoid = self.linear1(x).sigmoid()
        h2_sigmoid = self.linear2(h1_sigmoid).sigmoid()
        y_pred = self.linear3(h2_sigmoid)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, D_out = 200, x1_np.shape[1], 200, 50, 10

model = Net(D_in, H1, H2, D_out)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

T = 50 # number of training epoch

corr = np.corrcoef(x1, x2)[0:x1.shape[0],x1.shape[0]:(x1.shape[0]+x2.shape[0])] # define the correspondence matrix that suits your datasets
```

Then, the two networks are trained by `train_and_project` function:

```python
projections = train_and_project(x1_np, x2_np, model=model, optim=optimizer, T=50) # x1_np and x2_np are numpy arrays of two input modals
```

### Phase 2: Classification using Network of Feature Regularization

define your classification model:

```python
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 100)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 50)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(50,10)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
``` 

The regularized model is then trained with function `train_epoch`:

```python
from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(X,y,test_size=0.1,
                                                        random_state=10, stratify = y)    
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
     
net = Net()
opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()
e_losses = []
num_epochs = 20
for e in range(num_epochs):
    e_losses += train_epoch(net, X_train, y_train, opt, criterion, sim)
with torch.no_grad():
    x_tensor_test = torch.from_numpy(X_vali).float()#.to(device)
    net.eval()
    yhat = net(x_tensor_test)
y_pred_softmax = torch.log_softmax(yhat, dim = 1)
_, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
correct_pred = np.mean([float(y_pred_tags[i] == y_vali[i]) for i in range(len(y_vali))])
print("Round",i,"Test Accuracy (regularized):",correct_pred)
acc_reg.append(np.mean(correct_pred))  
```

## License
MIT License

Copyright (c) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
