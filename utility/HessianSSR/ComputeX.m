function [X] = ComputeX(Nodes)

[NumNodes,Dim] = size(Nodes);
CoordCombSize = sum([1:Dim])+Dim+1;

X = zeros(NumNodes,CoordCombSize);

for NIdx = 1:NumNodes
    RIdx = 1;
    for i=1:Dim
        for j=i:Dim
            X(NIdx,RIdx) = Nodes(NIdx,i)*Nodes(NIdx,j);
            RIdx = RIdx+1;
        end
    end

    for i=1:Dim
        X(NIdx,RIdx) = Nodes(NIdx,i);
        RIdx = RIdx+1;
    end

    X(NIdx,RIdx) = 1;
end
