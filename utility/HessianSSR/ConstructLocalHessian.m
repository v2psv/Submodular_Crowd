%%% this function constructs two operators B and BS based on fitting
%%% a quadratic polynomial in the given normal coordinate; note that the
%%% zeroth-order term of this polynomial is fixed

function [B,BS] = ConstructLocalHessian(Nodes, Base, BaseIndexInG)

TRUNCLIMIT = 0.1e-9;

[NumNodes,Dim] = size(Nodes);

[X] = ComputeX(Nodes);
[XBase] = ComputeX(Base);
X = X(:,1:end-1);
XBase = XBase(:,1:end-1);

XInv = pinv(X);

IndMat = zeros(NumNodes,NumNodes);
IndMat(:,BaseIndexInG) = 1;
RegMat = XInv-XInv*IndMat;

B = zeros(NumNodes,NumNodes);
RIdx = 1;
for i=1:Dim
    for j=i:Dim
        if i==j
            B = B+2*RegMat(RIdx,:)'*RegMat(RIdx,:);
        else
            B = B+RegMat(RIdx,:)'*RegMat(RIdx,:);
        end
        RIdx = RIdx+1;
    end
end

if nargout > 1
    %%% weighted penalization of the local deviation
        NodesNormSq = sum(Nodes.^2,2)';
        BaseNormSq = sum(Base.^2,2)';
        DistSq = Nodes*Base';
        DistSq = DistSq';
        DistSq = abs(NodesNormSq-2*DistSq+BaseNormSq);
        DistSq = sort(DistSq,'ascend');
        RParam = DistSq(end)+TRUNCLIMIT;
        [W] = CompactKernel(Nodes, Base, RParam);
        W = diag(W);
    %%% weighted penalization of the local deviation
    T = X*XInv-eye(size(W));
    T = T-T*IndMat;
    BS = T'*W*T;
end


