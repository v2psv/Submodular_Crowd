function LabeledIdx = SelectFrames(Pool, method, option)

  if strcmp(method, 'random')==1
    NPool = size(Pool.Feature, 1);
    index = randperm(NPool);
    LabeledIdx = option.PoolIndex(index(1:option.LabelNum));

  elseif strcmp(method, 'k-means')==1
    cluster = kmeans(Pool.Feature, option.LabelNum);

    LabeledIdx = [];
    for i = 1:option.LabelNum
        % CIdx = find(cluster==i);
        idx = datasample(find(cluster==i), 1);
        LabeledIdx = [LabeledIdx option.PoolIndex(idx)];
    end

    % LabeledIdx = [];
  	% for i = 1:option.LabelNum
  	% 	index = find(cluster==i);
  	% 	X = Pool.Feature(index,:);
  	% 	centre = mean(X, 1);
  	% 	dist = pdist2(X, centre);
  	% 	[~, id] = min(dist);
  	% 	LabeledIdx = [LabeledIdx index(id)];
  	% end

  elseif strcmp(method, 'm-landmark')==1
     index = Mlandmark(Pool.Feature, option.LabelNum);
     LabeledIdx = option.PoolIndex(index);

  elseif strcmp(method, 'submodular_mix')==1
    Sigma = (1-option.theta)*option.sigmaK(option.PoolIndex,option.PoolIndex) + option.theta*option.sigmaT(option.PoolIndex,option.PoolIndex);
    Cluster = SClustering(Sigma, option.Group);
    index = GreedyMix(Sigma, Cluster', option.Group, option.LabelNum, []);
    LabeledIdx = option.PoolIndex(index);
  end
end
