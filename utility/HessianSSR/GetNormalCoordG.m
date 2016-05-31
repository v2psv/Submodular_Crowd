%%% this function computes an estimate of normal coordinates of given data
%%% points (CtRef) centered at CtBase using PCA; note that the normal
%%% coordinate value of CtBase should be 0

function [NCoordsRef,TS] = GetNormalCoordG(CtBase, CtRef, TanParam)

% TRUNCLIMIT = 0.1e-9;
[NumRef,SizeSpRef] = size(CtRef);

%%% get direction vectors in normal coordinates
    %%% estimate tangent space
        [TS.Forward, TS.Backward, TS.MeanVector, EValues] = PerformPCA(CtRef);
        %%% estimate the dimensionality of tangent space
        if TanParam.DimGiven == 1;
            TS.NCoordDim = TanParam.NCoordDim;
        else
            CumEValues = cumsum(EValues)/sum(EValues);
            TS.NCoordDim = find(CumEValues>=TanParam.EValueTolerance, 1);
            if isempty(TS.NCoordDim)
                TS.NCoordDim = size(TS.Forward,1);
            end
        end
        if TS.NCoordDim <= 0
            TS.NCoordDim = 1;
        end
        %%% estimate the dimensionality of tangent space
    %%% estimate tangent space
    %%% project data onto the tangent space        
        [NCoordBase] = ForwardPCA(CtBase, TS.Forward, TS.MeanVector, TS.NCoordDim);
        TS.Base = NCoordBase;
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
