function Plot3DCoordsofImgs(Loc, LabelIdx, Colormap)

if nargin < 3
    plot3(Loc(:,1),Loc(:,2),Loc(:,3),'.');
    hold on;
    plot3(Loc(LabelIdx,1),Loc(LabelIdx,2),Loc(LabelIdx,3),'ro');
    axis([-1 1 -1 1]);
    axis equal;
else
    [NumPoints,DimPoints] = size(Loc);
    hold on;
    for i=1:NumPoints
        plot3(Loc(i,1),Loc(i,2),Loc(i,3),'.','MarkerEdgeColor',Colormap(i,:));
    end
    plot3(Loc(LabelIdx,1),Loc(LabelIdx,2),Loc(LabelIdx,3),'ro');
    axis([-1 1 -1 1]);
    axis equal;
    hold off;
end
