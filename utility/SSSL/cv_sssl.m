function [opt_sigma] = cv_sssl(xTrain, yTrain, s)
    [xTrain,xm,xs]=normalize(xTrain);
    [yTrain,ym,ys]=normalize(yTrain);
%     yTrain = yTrain - mean(yTrain);
    
    sigma = heuristic_median(xTrain);
    bandwidth = sigma * exp(-2:0.2:4);
    error = zeros(1, length(bandwidth));
    for i = 1 : length(bandwidth)
        sigma = bandwidth(i);
        [Eigvec, Eigval] = eig_decomposition(Gaussian_kernel(xTrain, xTrain, sigma), s);

        % Eigenvectors Nxs
        V = Eigvec; 
        % Eigenvalues sxs
        D = diag(Eigval);
        % Whole with label. Nxn
        tmp_err = 0.0;
        n_label = size(xTrain,1);
        parfor j=1:n_label
            %disp(j)
            ind = zeros(1, n_label);
            ind(j) = 1;
           
            itrain = ~ind;
            itest = ~itrain;
            KB = Gaussian_kernel(xTrain, xTrain(itrain,:), sigma);
            % Whole with test.  NxT
            KT = Gaussian_kernel(xTrain, xTrain(itest,:), sigma);
        
            %results
            PredictY = KT' * V  * inv(V' * KB * KB' * V + 1e-6*eye(s)) * V' * KB * yTrain(itrain,:);
            tmp_err = tmp_err + sum(sum((PredictY - yTrain(itest,:)).^2));
        end
        error(i) = tmp_err;
    end
    [min_error,idx] = min(error);
    opt_sigma = bandwidth(idx);
    fprintf('Optimal %f--%f\n', opt_sigma, min_error);
end

