function [bandwidth] = heuristic_median(X)
	if size(X,1) < 2
		disp('Sample is not enough!')
	end
	bandwidth = median(pdist(X));
end
