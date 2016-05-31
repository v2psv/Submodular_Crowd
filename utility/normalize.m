function [norData,datamean,datastd]=normalize(data,m,s,flag)
    % data : N x d
    N = size(data,1);
    if nargin<=1
        datamean = mean(data,1);
        datastd = std(data,0,1);
        norData = (data - repmat(datamean, N, 1))./(repmat(datastd,N,1));
    elseif nargin==3
        norData = (data - repmat(m, N, 1))./(repmat(s,N,1));
        datamean=[];
        datastd=[];
    elseif nargin==4 && strcmp(flag,'reverse')
       norData=data.*repmat(s,N,1)+repmat(m,N,1);
       datamean=[];
       datastd=[];
    else
        disp('input of normalize data is error.\n')
        norData=[];
        datamean=[];
        datastd=[];
    end
end