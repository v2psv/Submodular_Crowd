function [Eigvec, Eigval] = eig_decomposition(G, s)
	% Perform eigendecomposition of G
	[M, lambda] = eig(G);
    
	% Sort eigenvectors in descending order
	[lambda, ind] = sort(diag(lambda), 'descend');
    
    %z = cumsum(lambda)./sum(lambda);
    %dim = sum(z<=s);
    dim = s;
	Eigvec = M(:, ind(1:dim));
	Eigval = lambda(1:dim);
end