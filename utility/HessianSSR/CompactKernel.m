function [K, KDer, KHes] = CompactKernel(EvalPts, RefPts, RParam)

[EvalNum,EvalSize] = size(EvalPts);
[RefNum,RefSize] = size(RefPts);

Ip = eye(EvalSize,EvalSize);
K = zeros(EvalNum,RefNum);
NZFlag = ones(EvalNum,RefNum);
DistanceSq = zeros(EvalNum,RefNum);
EvalPtsNormSq = sum(EvalPts.^2,2)';
RefPtsNormSq = sum(RefPts.^2,2)';
for i=1:EvalNum,
    InnerProduct = EvalPts(i,:)*RefPts';
    DistanceSq(i,:) = abs(EvalPtsNormSq(i)+RefPtsNormSq-2*InnerProduct);
    ZIdx = find(DistanceSq(i,:)>= RParam);
    NZIdx = find(DistanceSq(i,:) < RParam);
    K(i,ZIdx)= 0;
    NZFlag(i,ZIdx) = 0;
    K(i,NZIdx) = exp(RParam./(DistanceSq(i,NZIdx)-RParam));
end

if nargout >= 2
    KDer = zeros(EvalSize, EvalNum,RefNum);
    for i=1:EvalNum,
        for j=1:RefNum
            if NZFlag(i,j) > 0
                KDer(:,i,j) = -K(i,j)*2*RParam*(EvalPts(i,:)-RefPts(j,:))/(DistanceSq(i,j)-RParam)^2;
            else
                KDer(:,i,j) = 0;
            end
        end
    end
    KDerTemp = cell(EvalSize,1);
    for i=1:EvalSize
        KDerTemp{i} = reshape(KDer(i,:,:),EvalNum,RefNum);
    end
    KDer = KDerTemp;
    
    if nargout >= 3
        KHes = zeros(EvalSize, EvalSize, EvalNum,RefNum);
        for i=1:EvalNum,
            for j=1:RefNum
                if NZFlag(i,j) > 0
                    KHes(:,:,i,j) = (4*(RParam^2+2*RParam*(DistanceSq(i,j)-RParam))*K(i,j)*(EvalPts(i,:)-RefPts(j,:))'*(EvalPts(i,:)-RefPts(j,:)))/(DistanceSq(i,j)-RParam)^4-2*RParam*K(i,j)*Ip/(DistanceSq(i,j)-RParam)^2;
                else
                    KHes(:,:,i,j) = 0;
                end
            end
        end
        KHesTemp = cell(EvalSize,EvalSize);
        for i=1:EvalSize
            for j=1:EvalSize
                KHesTemp{i,j} = reshape(KHes(i,j,:,:),EvalNum,RefNum);
            end
        end
        KHes = KHesTemp;
    end
end
