% function [LapFeature] = getLapFeature(X, id, similarity, cluster)
% 	% cluster: from 1...K
% 	% Ncluster = max(cluster);
% 	% tmp_cluster = zeros(size(cluster));
% 	% tmp_cluster(id) = cluster(id);
% 	% sigma = sqrt(similarity);

% 	% LapFeature = [];
% 	% for iCluster = 1:Ncluster
% 	%     index = find(cluster==iCluster);
% 	%     comb = nchoosek(index, 2);
% 	%     D = X(comb(:,1),:) - X(comb(:,2),:);
% 	%     % S = sigma(comb(:,1),comb(:,2));
% 	%     S = sigma((comb(:,2)-1)*size(sigma,1)+comb(:,1));
% 	%     LapFeature = [LapFeature; D.*repmat(S,1,size(D,2))];
% 	% end

% 	X = X(id,:);
% 	[n,p] = size(X);
% 	cluster = cluster(id);
% 	similarity = similarity(id, id);
% 	Ncluster = max(cluster);
% 	sigma = sqrt(similarity);

% 	LapFeature = [];
% 	for iCluster = 1:Ncluster
% 		index = find(cluster==iCluster);
% 		if (length(index) > 1)
% 			comb = nchoosek(index, 2);
% 			D = X(comb(:,1),:) - X(comb(:,2),:);
% 			S = sigma((comb(:,2)-1)*n+comb(:,1));
% 			LapFeature = [LapFeature; D.*repmat(S,1,p)];
% 		end
% 	end


% 	% LapFeature = [];
% 	% clu = zeros(size(cluster));
% 	% clu(id) = cluster(id);
% 	% Ncluster = max(clu);
% 	% sigma = sqrt(similarity);
% 	% for iCluster = 1:Ncluster
% 	%     index = find(clu==iCluster);
% 	%     for i=1:length(index)
% 	%         for j=i+1:length(index)
% 	%             LapFeature = [LapFeature; (X(i,:)-X(j,:))*sigma(index(i),index(j))];
% 	%         end
% 	%     end
% 	% end
% end

% function [LapFeature] = getLapFeature(X, id, similarity, k)

% 	X = X(id,:);
% 	[n,p] = size(X);
% 	similarity = similarity(id, id);
% 	sigma = sqrt(similarity);
% 	[sigma, id] = sort(sigma, 2, 'descend');

% 	k = min(size(X,1)-1, k);

% 	LapFeature = [];
% 	for i = 2:k+1
% 		LapFeature = [LapFeature; (X-X(id(:,i),:)).*repmat(sigma(:,i),1,p)];
% 	end
% end


% function [LapFeature] = getLapFeature(X, sigma, k)
% 	[n,p] = size(X);
% 	[sigma, id] = sort(sigma, 2, 'descend');

% 	k = min(size(X,1)-1, k);

% 	LapFeature = [];
% 	for i = 2:k+1
% 		LapFeature = [LapFeature; (X-X(id(:,i),:)).*repmat(sigma(:,i),1,p)];
% 	end
% 	LapFeature = LapFeature(any(LapFeature,2),:); % remove rows with all zeros
% end

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