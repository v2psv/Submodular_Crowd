function feaIndex = getFeatureIndex(features, datasetID)
    size = {[1 2],[1 2]};               % area, perimeter
    perimeter_orien = {[3:8],[3:6]};	% perimeter orientation
    ratio = {[9],[7]};
    edge = {[10],[8]};                  % edge
    edge_orien = {[11:16],[9:12]};      % edge orientation
    minkowski_dim = {[17],[13]};
    glcm = {[18:29],[14:25]};

    if datasetID == 4       % CVPR_new
        id = 1;
    elseif datasetID == 0   % Pedes0
        id = 2;
    elseif datasetID == 3   % Mall
        id = 3;
    elseif datasetID == 5   % fudan
        id = 1;
    end


    feaIndex = [];
    if strcmp(features,'all') == 1
        if datasetID == 0
            feaIndex = [1:29];
        elseif datasetID == 1
            feaIndex = [1:29];
        elseif datasetID == 2
            feaIndex = [1:18];
        elseif datasetID == 3
            feaIndex = [1:29];
        elseif datasetID == 4
            feaIndex = [1:29];
        elseif datasetID == 5
            feaIndex = [1:29];
        return;
    end;

    if ~isempty(strfind(features, 'S')) % size
        feaIndex = [feaIndex size{id}];
    end
    if ~isempty(strfind(features, 'P')) % perimeter  orientation
        feaIndex = [feaIndex perimeter_orien{id}];
    end
    if ~isempty(strfind(features, 'R')) % ratio
        feaIndex = [feaIndex ratio{id}];
    end
    if ~isempty(strfind(features, 'E')) % edge
        feaIndex = [feaIndex edge{id}];
    end
    if ~isempty(strfind(features, 'Y')) % edge orientation
        feaIndex = [feaIndex edge_orien{id}];
    end
    if ~isempty(strfind(features, 'M'))
        feaIndex = [feaIndex minkowski_dim{id}];
    end
    if ~isempty(strfind(features, 'G'))
        feaIndex = [feaIndex glcm{id}];
    end

end