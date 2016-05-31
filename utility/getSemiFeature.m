function [SemiFeature] = getSemiFeature(Feature, similarity, cluster)
    SemiFeature = [];
    Ncluster = max(cluster);
    sigma = sqrt(similarity);
    for iCluster = 1:Ncluster
        index = find(cluster==iCluster);
        % strip_indexI = zeros((length(index)-1)*length(index)/2,1);
        % strip_indexJ = zeros((length(index)-1)*length(index)/2,1);
     %    for i = index
     %    	for j = index
     %    		strip_indexI = [strip_indexI; i];
     %    		strip_indexJ = [strip_indexI; j];
     %    	end
     %    end
     %    tmp = zeros(length(strip_indexI),size(Feature,2));
     %    parfor id=1:length(strip_indexI)
     %    	i = strip_indexI(id);
     %    	j = strip_indexJ(id);
     %    	tmp(id,:) = (Feature(i,:)-Feature(j,:))*sigma(index(i),index(j));
    	% end
    	% SemiFeature = [SemiFeature; tmp];

        for i=1:length(index)
            for j=i+1:length(index)
                SemiFeature = [SemiFeature; (Feature(i,:)-Feature(j,:))*sigma(index(i),index(j))];
            end
        end
    end
end