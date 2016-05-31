function [NCoordsRef,TS] = GetNormalCoordGGivenTS(CtRef,TS)

% TRUNCLIMIT = 0.1e-9;
[NumRef,SizeSpRef] = size(CtRef);

%%% get direction vectors in normal coordinates
    %%% project data onto the tangent space
        [NCoordsRef] = ForwardPCA(CtRef, TS.Forward, TS.MeanVector, TS.NCoordDim);
    %%% project data onto the tangent space
    %%% center at base
        for i=1:NumRef
            NCoordsRef(i,:) = NCoordsRef(i,:)-TS.Base;
        end
    %%% center at base    
%%% get direction vectors in normal coordinates

% %%% calculate geodesic distances
%     GDist = zeros(1,NumRef);
%     for i=1:NumRef
%         GDist(i) = GeodesicDistinS(CtBase,CtRef(i,:),R);
%         if GDist(i)<TRUNCLIMIT
%             GDist(i) = 0;
%         end
%     end
% %%% calculate geodesic distances
% %%% resize vectors
%     for i=1:NumRef
%         if GDist(i)>0
%             NCoordsRef(i,:) = NCoordsRef(i,:)/norm(NCoordsRef(i,:),2)*GDist(i);
%         end
%     end
% %%% resize vectors
