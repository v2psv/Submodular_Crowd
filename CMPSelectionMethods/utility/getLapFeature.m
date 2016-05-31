function [LapFeature] = getLapFeature(X, sigma, k)
        if nargin < 3
                k = 10;
        end

        [n,p] = size(X);
        [sigma, id] = sort(sigma, 2, 'descend');

        k = min(size(X,1)-1, k);

        LapFeature = [];
        for i = 2:k+1
                LapFeature = [LapFeature; (X-X(id(:,i),:)).*repmat(sigma(:,i),1,p)];
        end
        LapFeature = LapFeature(any(LapFeature,2),:); % remove rows with all zeros
end