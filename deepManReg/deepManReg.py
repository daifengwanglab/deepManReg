import numpy as np
import scipy.spatial.distance as sd
from neighborhood import neighbor_graph, laplacian
from correspondence import Correspondence
from stiefel import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from datareader import *
import pandas as pd 
import os.path
import pdb
cuda = torch.device('cuda') 
import scipy as sp
from collections import Counter
import seaborn as sns
from random import sample
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt

def train_and_project(x1_np, x2_np, k, corr, model, opt, T):

    model = model

    x1 = torch.from_numpy(x1_np.astype(np.float32))
    x2 = torch.from_numpy(x2_np.astype(np.float32))
    print(x1.dtype)
    
    adj1 = neighbor_graph(x1_np, k=k)
    adj2 = neighbor_graph(x2_np, k=k)
    
    # correspond matrix (correlation + 5nn graph)
    w = np.block([[corr,adj1],
                  [adj2,ncorr.T]])

    L_np = laplacian(w, normed=False)
    L = torch.from_numpy(L_np.astype(np.float32))
    
    opt = optimizer
    
    for t in range(T):
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

        opt.zero_grad()
        loss.backward(retain_graph=True)

        # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
        rgrad = proj_stiefel(proj_outputs, proj_outputs.grad) 

        opt.zero_grad()
        # Backpropogate the Rimannian gradient w.r.t proj_outputs
        proj_outputs.backward(rgrad)

        opt.step()
        
    proj_outputs_np = proj_outputs.detach().numpy()
    return proj_outputs_np

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
