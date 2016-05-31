% This code shows a toy demo of regression based on Hessian energy. There are 3 parameters to be tuned for standard Hessian-based regression:
%  1. 'KNSize' specifies the size of nearest neighbors for each data point. This parameter affects the estimation of local tangent space and the estimation local Hessian.
%  2. 'TanParam.NCoordDim' specifies the dimensionality of tangent space for each data point
%  3. 'Lambda1' is the usual regularization parameter.

% In general, we recommend tuning the parameters using cross-validation based on given labeled data points. For details, please refer to our paper
%  Kwang In Kim, Florian Steinke and Matthias Hein, "Semi-supervised Regression using Hessian energy with an application to semi-supervised dimensionality reduction", Advances in Neural Information Processing Systems 22.

clear;
close all;

KNSize = 20;
TanParam.DimGiven = 1;
TanParam.NCoordDim = 1;

Lambda1 = 0.000010;

%%% generate data
    Theta = [2*pi:0.01:5*pi];
    R = Theta/5;
    NumTrn = length(Theta);
    
    Nodes = zeros(NumTrn,2);
    Labels.y = zeros(NumTrn,1);
    Labels.LFlag = zeros(NumTrn,1);
        
    for i=1:NumTrn
        RNoise = zeros(2,1);
        Nodes(i,1) = R(i)*cos(Theta(i))+RNoise(1);
        Nodes(i,2) = R(i)*sin(Theta(i))+RNoise(2);
    end

    LLoc = [200,NumTrn-200];
    Labels.y(LLoc(1)) = 10;
    Labels.y(LLoc(2)) = 20;
    Labels.LFlag(LLoc) = 1;
    
    subplot(1,3,1);
    plot3(Nodes(:,1),Nodes(:,2),Labels.y, '.-');
    hold on;
    plot3(Nodes(LLoc,1),Nodes(LLoc,2),Labels.y(LLoc), 'r.', 'MarkerSize',30);
    hold off;
    grid on;
    title('data')
%%% generate data

%% perform regression
    [NNIdx, KDistance] = GetKNN(Nodes, KNSize);
    [B] = ConstructHessian(Nodes, NNIdx, TanParam);
    [F] = PerformRegression(Labels, B, Lambda1);
%% perform regression

subplot(1,3,2);
plot3(Nodes(:,1),Nodes(:,2),F, '.-');
hold on;
plot3(Nodes([LLoc],1),Nodes([LLoc],2),Labels.y([LLoc]), 'r.', 'MarkerSize',30);
hold off;
grid on;
title('regression result')

%%% show eigenvectors
    GridSize = length(F);
    Grid = zeros(GridSize,1);
    Grid(1) = 0;
    for i=2:GridSize
        Grid(i) = Grid(i-1)+norm(Nodes(i,:)-Nodes(i-1,:));
    end
    [evec,eval] = eig(full(B));
    eval = diag(eval);
    eval(1:10)
    subplot(1,3,3);
    plot(Grid,evec(:,1),'r',Grid,evec(:,2),'g',Grid,evec(:,3),'b',Grid,evec(:,4),'m');
    legend('1st','2nd','3rd','4th');
    grid on;
    title('eigenvectors of B')
%%% show eigenvectors

display('done.');
