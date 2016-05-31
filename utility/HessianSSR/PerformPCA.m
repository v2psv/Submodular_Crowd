function [Forward, Backward, MeanVector, EValues] = PerformPCA(Data)

[NumPtns, OutputSize] = size(Data);

MeanVector = mean(Data,1);
for i=1:NumPtns
    Data(i,:) = Data(i,:)-MeanVector;
end

if NumPtns > OutputSize*2;
    Cov = cov(Data);
    [evecs,evals] = eig(Cov);
    evals = diag(evals)';

    if OutputSize > 1
        if evals(1)<evals(end)
            evecs = fliplr(evecs);
            evals = fliplr(evals);
        end
    end

    Forward = evecs';
    Backward = Forward';
    EValues = evals;
else
    ADimData = min(NumPtns,OutputSize);

    K = Data*Data';
    [evecs,evals] = eig(K);
    evals = diag(evals)';

    if OutputSize > 1
        if evals(1)<evals(end)
            evecs = fliplr(evecs);
            evals = fliplr(evals);
        end
    end

    for i=1:NumPtns
        if evals(i)>0
            evecs(:,i) = evecs(:,i)/sqrt(evals(i));
        end
    end

    Forward = evecs'*Data;
    Forward = Forward(1:ADimData,:);
    EValues = evals(1:ADimData)/(NumPtns-1);
    Backward = Forward';
end
