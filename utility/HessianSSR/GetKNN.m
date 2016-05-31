% [NNIdx, KDist] = GetKNN(Nodes, MaxKNNSize)
% INPUT:
% 'Nodes': data matrix of size N times d (N = number of data points, d = dimensionality of the data).
% 'MaxKNNSize': Number of nearest neighbors which should be computed
% OUTPUT:
% 'NNIdx': nearest neighbor index of size (N times MaxKNNSize).
% 'KDist': returns thhe KNN distances (N times MaxKNNSize)
% 
% computes ('MaxKNNSize'-1)-nearest neighbors for each data point 
% (the point itself is also included in the list)
% and returns for each datapoint the indices of its nearest neighbors.
% For large scale problems, one might have to replace this function by an
% approximate nearest neighbor search module

function [StoredIdx, KDistance, Input, Stored] = GetKNN(Input, K, Stored)

if nargin >= 3
    [NumTstPtns,TstInputSize] = size(Input);
    [NumStoredPtns,StoredInputSize] = size(Stored);
    if TstInputSize==StoredInputSize
        StoredIdx = zeros(NumTstPtns,K);
        KDistance = zeros(NumTstPtns,K);
        StoredNormSq = sum(Stored.^2,2)';
        TstNormSq = sum(Input.^2,2)';
        
        for i=1: NumTstPtns
            InnerProduct = Input(i,:)*Stored';
            DistanceSq = abs(StoredNormSq-2*InnerProduct+TstNormSq(i));
            Distance = sqrt(DistanceSq);
            if K>1
                [Sorted, KIndex] = sort(Distance, 'ascend');
                StoredIdx(i,:) = KIndex(1:K);
                KDistance(i,:) = Sorted(1:K);
            else
                KDistance(i) = min(Distance);
                Temp = find(Distance==KDistance(i));
                StoredIdx(i) = Temp(1);
            end
        end
    else
        display('Model incompatible..');
    end
else    
    [NumTstPtns,TstInputSize] = size(Input);
    StoredIdx = zeros(NumTstPtns,K);
    KDistance = zeros(NumTstPtns,K);
    TstNormSq = sum(Input.^2,2)';
    for i=1: NumTstPtns
        InnerProduct = Input(i,:)*Input';
        DistanceSq = abs(TstNormSq-2*InnerProduct+TstNormSq(i));
        Distance = sqrt(DistanceSq);
        if K>1
            [Sorted, KIndex] = sort(Distance, 'ascend');
            StoredIdx(i,:) = KIndex(1:K);
            KDistance(i,:) = Sorted(1:K);
        else
            KDistance(i) = min(Distance);
            Temp = find(Distance==KDistance(i));
            StoredIdx(i) = Temp(1);
        end
    end
end
