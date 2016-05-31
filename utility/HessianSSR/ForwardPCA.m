function [Feature] = ForwardPCA(Input, Forward, Mean, LatDim)

[FSize, ISize] = size(Forward);
[INum, InputSize] = size(Input);

if ISize ~= InputSize
    display('PCA model incompatible..');
    return;
end

Forward = Forward(1:LatDim,:);

CenteredInput = (Input-repmat(Mean,INum, 1));
Feature = CenteredInput*Forward';

% clear CenteredInput;
