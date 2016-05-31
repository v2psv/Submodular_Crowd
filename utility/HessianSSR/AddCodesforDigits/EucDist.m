function [Dist] = EucDist(Input, Base)

[NumTrnPtns,TrnInputSize] = size(Input);
[NumBasePtns,BaseSize] = size(Base);
Dist = zeros(NumBasePtns,NumTrnPtns);
DistanceSq = zeros(1, NumTrnPtns);
InnerProduct = zeros(1, NumTrnPtns);
InputNormSq = sum(Input.^2,2)';
BaseNormSq = sum(Base.^2,2)';
for i=1:NumBasePtns,
    PatternNormSq = BaseNormSq(i);
    InnerProduct = Base(i,:)*Input';
    DistanceSq = abs(PatternNormSq+InputNormSq-2*InnerProduct);
    Dist(i,:) = sqrt(DistanceSq);
end
clear Distance InnerProduct LineNormSq LineNorm PatternNorm PatternNormSq;

