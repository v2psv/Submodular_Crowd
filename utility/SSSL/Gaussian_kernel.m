
function [G] = Gaussian_kernel(X, Y, sigma)
	% X : m x d
	% Y : n x d
	% G : m x n
	D = pdist2(X, Y, 'euclidean');
	G = exp(-1/( 2 * sigma^2) .* D.^2 );

end

