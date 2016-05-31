function [L] = Mlandmark(X, k)
	neighbor_num = 20;
	D = pdist2(X,X).^2;
	[~,W,~] = scale_dist(D, floor(neighbor_num/2));
	% calculate degree matrix
	degs = sum(W, 2);
	D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

	L = D - W;
	degs(degs == 0) = eps;
	D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
	L = D * L * D;

	% diff   = eps;
	[U,~] = eigs(L, k);
	% U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
	[cluster] = kmeans(U, k);

	L = [];
	for i=1:k
		index = find(cluster==i);
		s = X(index,:);
		centre = mean(s, 1);
		dist = pdist2(s, centre);
		[~, id] = min(dist);
		L = [L index(id)];
	end
end
