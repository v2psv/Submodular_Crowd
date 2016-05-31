This is code for the semi-supervised regression using Hessian energy:

Kwang In Kim, Florian Steinke and Matthias Hein, 
"Semi-supervised Regression using Hessian energy with an application 
to semi-supervised dimensionality reduction"
NIPS 2009

When you use the code, please cite the above paper.


Usage:
There are three main program components for the regression system, described below. 
Example usage can be found in 'ExampleSinusoidsonSpiral.m', 'ExampleDigits.m', 
and 'ExampleTwoPointsonSpiral.m'.

Standard regression uses functions 1-3 described below. 
Although we do not necessarily recommend it (see the paper), it is possible to 
use additional stabilizer which is described in the supplementary material of the 
corresponding paper. In this case, please see 1, 2-1, and 3-1.

1. [NNIdx, KDist] = GetKNN(Nodes, MaxKNSize)
   computes ('MaxKNNSize' - 1) -nearest neighbors of each data point stored in 'Nodes'.

   INPUT:
    'Nodes': data matrix of size N times d (N = number of data points, d = dimensionality of the data).
    'MaxKNNSize': Number of nearest neighbors which should be computed
   OUTPUT:
    'NNIdx': nearest neighbor index of size (N times MaxKNNSize).
    'KDist': returns thhe KNN distances (N times MaxKNNSize)

   For a detailed description use help GetKNN



2. [B] = ConstructHessian(Nodes, NNIdx, TanParam)
   constructs Hessian operator B such that <f,Bf> is an estimate of the 
   Hessian energy of f (see paper)
   
   INPUT:
    'Nodes': (N times d) data matrix; 'N': number of data points, 'd': 
              dimensionality of data.
    'NNIdx': (N times k) matrix containing indices of k-nearest neighbors for
             each data point (This matrix can be obtained using GetKNN)
    'TanParam': an integer specifying the dimensionality of the manifold
              (for SSR we recommend to do cross-validation over this parameter if
              the dimensionality of the manifold is unknown)
   OUTPUT:
    'B':  (N times N) matrix - Hessian operator B as described in the paper
    'BS': (N times N) matrix - BS penalizes deviation of the function from
                               its second order approximation
   USAGE: B = ConstructHessian(Nodes, NNIdx, TanParam)
          computes *only* the Hessian operator B (saves time and memory
          compared so the following second option)
 
          [B, BS] = ConstructHessian(Nodes, NNIdx, TanParam)
          computes B and BS 

   For a detailed description use help ConstructHessian



3. [f] = PerformRegression(Labels, B, Lambda)
   Performs semi-supervised regression with the squared loss using the Hessian 
   operator B (see ConstructHessian) as regularizer -    

   INPUT:
    'B': (N times N) regularization matrix calculated using 'ConstructHessian.m'
         such that the Hessian energy of f is approximated as <f,Bf>.
    'Labels.LFlag' 
         if Labels.LFlag(i)=1, 'i'-th node is a labeled point.
         if Labels.LFlag(i)=0, 'i'-th node is not a labeled point.
    'Labels.y' = real column vector of size N containing labels. For the locations,
         where the corresponding values of 'Labels.LFlag' are not 1,
         the values of 'Labels.y' do not affect the regression.
   OUTPUT: 
    'f':  learned function on the N points 
  
    performs regression by minimizing the squared loss (only on the labeled
    points) plus Hessian energy <f,Bf> weighted by Lambda.
   
    optionally, [f] = PerformRegression(Labels, B, Lambda1, BS, Lambda2) 
    performs regression by minimizing the squared (only on the labeled
    points) plus stabilized Hessian energy <f,(B+(Lambda2/Lambda1)*BS)f> weighted by Lambda1.
    'BS': (N times N) matrix-- can be computed using ConstructHessian.m
