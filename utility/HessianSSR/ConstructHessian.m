% [B,BS] = ConstructHessian(Nodes, NNIdx, TanParam)
% INPUT:
% 'Nodes': (N times d) data matrix; 'N': number of data points, 'd': 
%           dimensionality of data.
% 'NNIdx': (N times k) matrix containing indices of k-nearest neighbors for
%           each data point (This matrix can be obtained using GetKNN)
% 'TanParam': an integer specifying the dimensionality of the manifold
%             (for SSR we recommend to do cross-validation over this parameter if
%             the dimensionality of the manifold is unknown)
% OUTPUT:
% 'B': (N times N) matrix - Hessian operator B as described in the paper
% 'BS': (N times N) matrix - BS penalizes deviation of the function from
%                            its second order approximation
% USAGE: B = ConstructHessian(Nodes, NNIdx, TanParam)
%        computes *only* the Hessian operator B (saves time and memory
%        compared so the following second option)
%
%        [B, BS] = ConstructHessian(Nodes, NNIdx, TanParam)
%        computes B and BS 
%
% This function constructs a regularization operator 'B' such that the
% Hessian energy of a function f is approximated as <f,Bf>.
% 
% Optionally, one can construct additionally the stabilization operator 'BS' which
% penalizes the deviation of f from its second order approximation
% In rare cases B does not behave well - in this case <f,(B + gamma*BS) f>,
% for small gamma is a better estimator of the Hessian.
% In general we recommend using only 'B'.

function [B,BS] = ConstructHessian(Nodes, NNIdx, TanParam)

[NumNodes,AmbDim] = size(Nodes);

BlSpMatSize = 300000;
MaxNumNN = size(NNIdx,2);
NumNonzero = NumNodes*MaxNumNN^2;
BlSpMatNum = ceil(NumNonzero/BlSpMatSize);
BlSpB = cell(BlSpMatNum,1);

SpMatIdx = 1;
SpEleCounter = 1;
SpRowIdx = zeros(BlSpMatSize,1);
SpColIdx = zeros(BlSpMatSize,1);
BVal = zeros(BlSpMatSize,1);

if nargout > 1
    BlSpBS = cell(BlSpMatNum,1);
    BSVal = zeros(BlSpMatSize,1);
end

for i=1:NumNodes
    G = Nodes(NNIdx(i,:),:);
    
    %%% compute normal coordinates of data points which are in the neighborhood of i-th data point
        [NumNeighbors,AmbDim] = size(G);
        CtBase = Nodes(i,:);
        [NG,TS] = GetNormalCoordG(CtBase, G, TanParam);
        [NBase,TS] = GetNormalCoordGGivenTS(CtBase,TS);
        BaseIndexInG = find(NNIdx(i,:)==i,1);
    %%% compute normal coordinates of data points which are in the neighborhood of i-th data point
    
    %%% compute local Hessian and stabilizer based on local polynomial fitting in the normal coordinate
        if nargout > 1
            [LocalB, LocalBS] = ConstructLocalHessian(NG, NBase, BaseIndexInG);
        else
            [LocalB] = ConstructLocalHessian(NG, NBase, BaseIndexInG);
        end
    %%% compute local Hessian and stabilizer based on local polynomial fitting in the normal coordinate
    for j=1:NumNeighbors
        GRIdx = NNIdx(i,j);

        GCIdx = NNIdx(i,1:NumNeighbors);
        SpRowIdx(SpEleCounter:SpEleCounter+NumNeighbors-1) = GRIdx;
        SpColIdx(SpEleCounter:SpEleCounter+NumNeighbors-1) = GCIdx;
        BVal(SpEleCounter:SpEleCounter+NumNeighbors-1) = LocalB(j,1:NumNeighbors);
        if nargout > 1
            BSVal(SpEleCounter:SpEleCounter+NumNeighbors-1) = LocalBS(j,1:NumNeighbors);
        end
        SpEleCounter = SpEleCounter+NumNeighbors;
    end
    if SpEleCounter > BlSpMatSize-NumNeighbors-1
        NonZeroIdx = length(find(SpRowIdx>0));
        SpRowIdx = SpRowIdx(1:NonZeroIdx);
        SpColIdx = SpColIdx(1:NonZeroIdx);
        BVal = BVal(1:NonZeroIdx);
        BlSpB{SpMatIdx} = sparse(SpRowIdx,SpColIdx,BVal,NumNodes,NumNodes);
        BVal = zeros(BlSpMatSize,1);
        if nargout > 1
            BSVal = BSVal(1:NonZeroIdx);
            BlSpBS{SpMatIdx} =sparse(SpRowIdx,SpColIdx,BSVal,NumNodes,NumNodes);
            BSVal = zeros(BlSpMatSize,1);
        end
        SpRowIdx = zeros(BlSpMatSize,1);
        SpColIdx = zeros(BlSpMatSize,1);
                
        SpMatIdx = SpMatIdx+1;
        SpEleCounter = 1;
    end
end

if SpEleCounter > 1
    NonZeroIdx = length(find(SpRowIdx>0));
    SpRowIdx = SpRowIdx(1:NonZeroIdx);
    SpColIdx = SpColIdx(1:NonZeroIdx);
    BVal = BVal(1:NonZeroIdx);
    BlSpB{SpMatIdx} = sparse(SpRowIdx,SpColIdx,BVal,NumNodes,NumNodes);
    if nargout > 1
        BSVal = BSVal(1:NonZeroIdx);
        BlSpBS{SpMatIdx} =sparse(SpRowIdx,SpColIdx,BSVal,NumNodes,NumNodes);        
    end

    SpMatIdx = SpMatIdx+1;
end

B = sparse([],[],[],NumNodes,NumNodes,0);
for i=1:SpMatIdx-1
    B = B+BlSpB{i};
end
B = (B+B')/2;

if nargout > 1
    BS =sparse([],[],[],NumNodes,NumNodes,0);
    for i=1:SpMatIdx-1
        BS = BS+BlSpBS{i};
    end
    BS = (BS+BS')/2;
end
