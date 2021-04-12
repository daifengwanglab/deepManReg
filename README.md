# deepManReg: a deep manifold-regularized learning model for improving phenotype prediction from multi-modal data

## Summary
The phenotypes of complex biological systems are fundamentally driven by various multi-scale mechanisms. Increasing multi-modal data enables a deeper understanding of underlying complex mechanisms across scales for the phenotypes such as single cell multi-omics data for cell types. However, integrating and interpreting such large-scale multi-modal data remains challenging, especially given highly heterogeneous, nonlinear relationships across modalities. To address this, we developed an interpretable regularized learning model, deepManReg to predict phenotypes from multi-modal data. First, deepManReg employs deep neural networks to learn cross-modal manifolds and then align multi-modal features onto a common latent space. This space aims to preserve both global consistency and local smoothness across modalities and to reveal higher-order nonlinear cross-modal relationships. Second, deepManReg uses cross-modal manifolds as a feature graph to regularize the classifiers for improving phenotype predictions and also prioritizing the multi-modal features and cross-modal interactions for the phenotypes. We applied deepManReg to (1) the image data of handwritten digits with multi-features and (2) recent single cell multi-modal data (Patch-seq data) including transcriptomics and electrophysiology for neuronal cells in the mouse brain. After comparing with the state-of-the-arts, we show that deepManReg has significantly improved predicting phenotypes in both datasets, and also prioritized genes and electrophysiological features for the phenotypes of neuronal cells 

## Flow chart
![alt text](https://github.com/daifengwanglab/deepManReg/figures/drawing.eps)

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

A demo for aligning and aligning single-cell multi-modal data is available at `/visual_sample/phase1.ipynb` and `/visual_sample/phase2.ipynb` for a Patch-seq dataset (3654 cells) in the mouse visual cortex (https://github.com/berenslab/layer4). The input data includes the gene expression levels and electrophysiological features of these cells in a rda file. Also, we provide a list of neuronal marker genes as select gene features. The code performs alignment on a reduced-dimensional space on single-cell electrophysiological data and gene expression with three major methods: deepManReg, Linear manifold alignment (NMA), Canonical correspondence analysis (CCA) and MATCHER.  The expected output includes the visualization of the latent space after each alignment method. The total running time of this demo was a couple of hours on a local laptop.

Others plots can be generated through the following codes files: `igraph_visual.Rmd` visualizes the gene regulatory network by igraph for each cluster; `boxplot_visual.Rmd` visualizes the boxplot comparing classification results between deepManReg and other methods, e.g., CCA, Linear Manifold Alignment, MATCHER.

## Usage

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
```

Then, the two networks are trained by `train_and_project` function:

```python
def train_and_project(x1_np, x2_np):
    
    torch.manual_seed(0)

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = 200, x1_np.shape[1], 200, 50, 10

    model = Net(D_in, H1, H2, D_out)

    x1 = torch.from_numpy(x1_np.astype(np.float32))
    x2 = torch.from_numpy(x2_np.astype(np.float32))
    print(x1.dtype)
    
    adj1 = neighbor_graph(x1_np, k=5)
    adj2 = neighbor_graph(x2_np, k=5)
    
    # correspond matrix (correlation + 5nn graph)
    w = np.block([[np.corrcoef(x1, x2)[0:x1.shape[0],x1.shape[0]:(x1.shape[0]+x2.shape[0])],adj1],
                  [adj2,np.corrcoef(x1, x2)[0:x1.shape[0],x1.shape[0]:(x1.shape[0]+x2.shape[0])].T]])

    L_np = laplacian(w, normed=False)
    L = torch.from_numpy(L_np.astype(np.float32))
    
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y1_pred = model(x1)
        y2_pred = model(x2)

        outputs = torch.cat((y1_pred, y2_pred), 0)
        
        # Project the output onto Stiefel Manifold
        u, s, v = torch.svd(outputs, some=True)
        #u, s, v = torch.svd(outputs+1e-4*outputs.mean()*torch.rand(outputs.shape[0],outputs.shape[1]), some=True)
        proj_outputs = u@v.t()

        # Compute and print loss
        print(L.dtype)
        loss = torch.trace(proj_outputs.t()@L@proj_outputs)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        proj_outputs.retain_grad()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
        rgrad = proj_stiefel(proj_outputs, proj_outputs.grad) 

        optimizer.zero_grad()
        # Backpropogate the Rimannian gradient w.r.t proj_outputs
        proj_outputs.backward(rgrad)

        optimizer.step()
        
    proj_outputs_np = proj_outputs.detach().numpy()
    return proj_outputs_np
    
projections = train_and_project(x1_np, x2_np) # x1_np and x2_np are numpy arrays of two input modals
```

### Phase 2: Classification using Network of Feature Regularization

The classification is then trained with function `train_epoch`:

```python
def train_epoch(model, X_train, y_train, opt, criterion, sim, batch_size=200):
    model.train()
    sim = sim
    losses = []
    for beg_i in range(0, X_train.size(0), batch_size):
        x_batch = X_train[beg_i:beg_i + batch_size, :]
        y_batch = y_train[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch.float())
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        #print(loss)
        reg = torch.tensor(0., requires_grad=True)
        for name, param in net.fc1.named_parameters():
            if 'weight' in name:
                M = .5 * ((torch.eye(feature_dim) - sim).T @ (torch.eye(feature_dim) - sim)) + .5 * torch.eye(feature_dim)
                reg = torch.norm(reg + param @ M.float() @ param.T, 2)
                loss += reg
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.numpy())
    return losses
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
