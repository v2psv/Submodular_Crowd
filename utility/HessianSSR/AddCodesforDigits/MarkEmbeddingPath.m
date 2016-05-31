function [TIdx,Loc,RefIdx] = MarkEmbeddingPath(Data, DBegin, DEnd, NumEvals, RefIdx)

[NumPoints,SizePoint] = size(Data);

if nargin < 5
    DMin = DBegin;
    DMax = DEnd;

    RD = DMax-DMin;

    Interval = RD/(NumEvals-1);
    
    Loc = zeros(NumEvals,SizePoint);
    Loc(1,:) = DMin;
    for i=1:NumEvals-1
        Loc(i+1,:) = Loc(i,:)+Interval;
    end
    
    if SizePoint == 2
        plot(Loc(:,1),Loc(:,2),'.');
    elseif SizePoint == 3
        plot(Loc(:,1),Loc(:,2),Loc(:,3),'.');
    end
    
    RefIdx = zeros(2,1);
    Dist = EucDist(Loc(1,:),Data);
    RefIdx(1) = find(Dist==min(Dist),1);
    Dist = EucDist(Loc(end,:),Data);
    RefIdx(2) = find(Dist==min(Dist),1);
else
    RD = Data(RefIdx(2),:)-Data(RefIdx(1),:);
    
    Interval = RD/(NumEvals-1);
    Loc = zeros(NumEvals,SizePoint);
    Loc(1,:) = Data(RefIdx(1),:);
    for i=1:NumEvals-1
        Loc(i+1,:) = Loc(i,:)+Interval;
    end
    if SizePoint == 2
        plot(Loc(:,1),Loc(:,2),'.g');
    elseif SizePoint == 3
        plot(Loc(:,1),Loc(:,2),Loc(:,3),'g.');
    end
end

TLoc = zeros(NumEvals,SizePoint);
TIdx = zeros(1,2);
for i=1:NumEvals
    Dist = EucDist(Loc(i,:),Data);
    Idx = find(Dist==min(Dist));
    TLoc(i,:) = Data(Idx,:);
    TIdx(i) = Idx;
end
if SizePoint == 2;
    plot(Loc(:,1),Loc(:,2),'r.',TLoc(:,1),TLoc(:,2),'bo','LineWidth',2);
    hold on;
    plot(TLoc([NumExtEvals+1,end-NumExtEvals],1),TLoc([NumExtEvals+1,end-NumExtEvals],2),'bo','MarkerEdgeColor', 'g', 'MarkerFaceColor', 'b', 'LineWidth',2);
    hold off;
    hold on;
    plot(Data(:,1),Data(:,2),'g.');
    hold off;
    axis equal;
end
