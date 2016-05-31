%  This code shows a demo of regression based on Hessian energy corresponding 
%  to Fig. 1 of the supplementary material of our paper.
%  Kwang In Kim, Florian Steinke and Matthias Hein, 
%  "Semi-supervised Regression using Hessian energy with an application 
%  to semi-supervised dimensionality reduction", NIPS 2009.
%
%  Please note that this code requires a data file containing the digit database
%  'Digits.mat' which can be downloaded from
%  http://www.ml.uni-saarland.de/code/HessianSSR/HessianSSR.html

%  There are 3 parameters to be tuned for regression:
%  1. 'KNSize' specifies the size of nearest neighbors for each data point. 
%     This parameter affects the estimation of local tangent space and the estimation local Hessian.
%  2. 'TanParam.NCoordDim' specifies the dimensionality of the manifold
%     resp. the dimension of the manifold
%  3. 'Lambda1' is the regularization parameter.

% In general, we recommend tuning the parameters using cross-validation based on given labeled data points.
% To facilitate the demo here, we provide a set of pre-optimized parameters.
% Please note that this demo takes several minutes to finish.

clear;
close all;

display('This demo takes some minutes...');
addpath './AddCodesforDigits';

load Digits.mat;
Labels.LFlag = zeros(size(Labels.y(:,1)));
Labels.LFlag(LLoc) = 1;

% % load DigitsEmbedding.mat;
% %% Perform regression
    [NumNodes, AmbDim] = size(Nodes);
    display('constructing k-nearest neighbor graph...');
    [NNIdx, KDistance] = GetKNN(Nodes, 40);
    save DigitsNNIdx.mat NNIdx;
    load DigitsNNIdx.mat NNIdx;
    EmbeddingCoord = zeros(NumNodes,4);
    ILabels.LFlag = Labels.LFlag;
    for i=1:4
        ILabels.y = Labels.y(:,i);
        display(sprintf('processing label%d',i));
        display('constructing Hessian operator...');
        [B] = ConstructHessian(Nodes, NNIdx(:,1:Param{i}.KNSize), Param{i}.TanParam);
        display('performing regression...');
        [EmbeddingCoord(:,i)] = PerformRegression(ILabels, B, Param{i}.Lambda1);
    end
%% Perform regression

save DigitsEmbedding.mat EmbeddingCoord;
load DigitsEmbedding.mat;

%%% display results
    ImgSize = sqrt(AmbDim);
    NumEvalPoints = 21;
    
    %%% normalize GT labels into [0,1]: for display only
        NormalBase = min(Labels.y);
        NormalScale = max(Labels.y)-min(Labels.y);
        NLabels.y = Labels.y;
        for i=1:NumNodes
            NLabels.y(i,:) = NLabels.y(i,:)-NormalBase;
            NLabels.y(i,:) = NLabels.y(i,:)./NormalScale;
        end
        NBegin = [0,0,0,0];
        NEnd = [1,1,1,1];
    %%% normalize GT labels into [0,1]: for display only

    %%% display a sampling of 4-dimensional parameterization from [0,0,0,0] to [1,1,1,1]
        %%% GT
            [GTLocIdx,Loc,RefLoc] = MarkEmbeddingPath(NLabels.y, NBegin, NEnd, NumEvalPoints);
            [GTLocIdx] = MarkEmbeddingPath(NLabels.y, NBegin, NEnd, NumEvalPoints, RefLoc);
        %%% GT
        %%% Hessian-based SSR
            NEmbeddingCoord = EmbeddingCoord;
            %%% normalize estimated coordinates based on the scale of GT
            %%% labels: for display only
                for i=1:NumNodes
                    NEmbeddingCoord(i,:) = NEmbeddingCoord(i,:)-NormalBase;
                    NEmbeddingCoord(i,:) = NEmbeddingCoord(i,:)./NormalScale;
                end
            %%% normalize estimated coordinates based on the scale of GT
            %%% labels: for display only
            [HLocIdx] = MarkEmbeddingPath(NEmbeddingCoord, NBegin, NEnd, NumEvalPoints, RefLoc);            
        %%% Hessian-based SSR
        figure;
        subplot(2,1,1);
        DisplayPatches(Nodes(GTLocIdx,:),length(GTLocIdx),ImgSize,ImgSize,length(GTLocIdx));
        colormap gray;
        title('ground truth');
        axis off;
        subplot(2,1,2);
        DisplayPatches(Nodes(HLocIdx,:),length(HLocIdx),ImgSize,ImgSize,length(HLocIdx));
        colormap gray;
        title('Hessian-based semi-supervised regression');
        axis off;
        colormap gray;
    %%% display a sampling of 4-dimensional parameterization from [0,0,0,0] to [1,1,1,1]
%%% display results

display('done.');
