function DisplayPatches(Data, DisplayNumber, PatchHeight, PatchWidth, ColSize)

[PatchNum, PatchSize] = size(Data);

if nargin < 5
    ColSize = floor(sqrt(PatchNum));
end

RowSize = floor(PatchNum/ColSize);

if PatchSize ~= PatchHeight*PatchWidth
    display('Patch size incompatible with data...');
    return;
end

if PatchNum > DisplayNumber
    PatchNum = DisplayNumber
end

% display patches
% DispPatch = zeros(PatchNum, PatchHeight+2, PatchWidth+2);
DispPatch = ones(PatchNum, PatchHeight+2, PatchWidth+2)*128;
for i=1: PatchNum,
    DispPatch(i,2:end-1,2:end-1) = reshape(Data(i,:),PatchHeight,PatchWidth);
end

%     MinVal = min(min(min(DispPatch)));
%     MaxVal = max(max(max(DispPatch)));
for i=1: PatchNum;
    MinVal = min(min(DispPatch(i,2:end-1,2:end-1)));
    MaxVal = max(max(DispPatch(i,2:end-1,2:end-1)));
    if (MaxVal-MinVal)>0 
        DispPatch(i,2:end-1,2:end-1) = (DispPatch(i,2:end-1,2:end-1)-MinVal)/(MaxVal-MinVal)*255;
    else
        DispPatch(i,:,:) = 0;
    end
end

A = squeeze(DispPatch(1,:,:));
for j=2: ColSize
    B = squeeze(DispPatch(j,:,:));
    A = cat(2,A,B);
end
C = A;
clear A;
image(C);
for i=1: RowSize-1
    A = squeeze(DispPatch((i*ColSize)+1,:,:));
    for j=2: ColSize
        B = squeeze(DispPatch((i*ColSize)+j,:,:));
        A = cat(2,A,B);
    end 
    C = cat(1,C,A);
    clear A;
end
imagesc(C);
clear DispPatch;
axis image;
