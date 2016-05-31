function [PredictY] = SSSL(xTrain, yTrain, xTest, N_label, s, sigma)
	% Implement the algorithm of 'A simple Algorithm for Semi-supervised Learning'
	% xTrain : N x D
	% yTrain : N x M
	% N_label : The first N_label YTrain for labeling
	% Xtest : N x D
	% s : Top s eigenfunction are used here.
	% PredictY : N x M
	%
    
	if nargin < 3
		disp('Not enough input arguments!') 
	end
	if nargin < 4
		N_label = 13;
	end
	if nargin < 5
		s = 20;
    end
     %normalize data
    [xTrain,xm,xs]=normalize(xTrain);
    xTest=normalize(xTest,xm,xs);
    
	xLabel = xTrain(1:N_label, :);
	yLabel = yTrain(1:N_label, :);
    [yLabel,ym,ys]=normalize(yLabel);
%     ym = mean(yLabel); yLabel = yLabel - ym;
    
	[Eigvec, Eigval] = eig_decomposition(Gaussian_kernel(xTrain, xTrain, sigma), s);

	% Eigenvectors Nxs
	V = Eigvec; 
	% Eigenvalues sxs
	D = diag(Eigval);
	% Whole with label. Nxn
	KB = Gaussian_kernel(xTrain, xLabel, sigma);
	% Whole with test.  NxT
	KT = Gaussian_kernel(xTrain, xTest, sigma);

	%results
	PredictY = KT' * V  * inv(V' * KB * KB' * V + 1e-6*eye(s)) * V' * KB * yLabel;
    PredictY=normalize(PredictY,ym,ys,'reverse');
%     PredictY = PredictY + ym;
end






