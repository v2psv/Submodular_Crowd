function [L,scores] = GreedyMix(sigma, cluster, Group, B, init, type)
    % sigma: affinity matrix
    % cluster: clusters
    % Group: number of clusters
    % B: budget
    % init: initial set
    if nargin < 6
        type = 'mix';
    end
    if nargin < 5
        L = [];
    else
        L = init;
    end

    if strcmp(type, 'mix')==1
        [L, scores] = mix_submodular(sigma, cluster, Group, B, L);
    elseif strcmp(type, 'locfac')==1
        [L,scores] = locfac_submodular(sigma, cluster, Group, B, L);
    elseif strcmp(type, 'diversity')==1
        [L,scores] = div_submodular(sigma, cluster, Group, B, L);
    end
end

function [L,scores] = mix_submodular(sigma, cluster, Group, B, L)
    N = length(cluster);
    scores = zeros(B,1);  % keep track of statistics
    weight = zeros(B,1);

    F = cell(Group, 1);
    for k = 1:Group
        idx = find(cluster==k);
        F{k}.cluster = idx;
        F{k}.fac = 0;
        F{k}.V = idx;
        F{k}.L = [];
        F{k}.N = length(idx);
        weight = F{k}.N;
    end
    Reward = zeros(Group,1);
    Number = ones(Group,1);

    bestId = 0;
    candidate = zeros(Group,1);
    for t = 1:B
        % find facility location candidates for each cluster
        if bestId == 0
            parfor k = 1:Group
                if length(F{k}.V) ~= 0
                    facility = zeros(1, length(F{k}.V)); % facility value for each left element in V
                    for i = 1:length(F{k}.V)
                        el = F{k}.V(i);
                        temp = [F{k}.L; el];
                        facility(i) = sum(max(sigma(temp,F{k}.cluster))) - F{k}.fac;
                    end
                    [~,id] = max(facility);
                    candidate(k) = F{k}.V(id);
                end
            end
        else
            if length(F{bestId}.V) ~= 0
                facility = zeros(1, length(F{bestId}.V)); % facility value for each left element in V
                for i = 1:length(F{bestId}.V)
                    el = F{bestId}.V(i);
                    temp = [F{bestId}.L; el];
                    facility(i) = sum(max(sigma(temp,F{bestId}.cluster)));
                end
                [~,id] = max(facility);
                candidate(bestId) = F{bestId}.V(id);
            end
        end

        % find optimal element that yields maximal diversity value
        rew = zeros(Group, 1);
        parfor k = 1:Group
            if candidate(k) ~= 0
                el = candidate(k);
                temp = Reward;
                temp(k) = temp(k) + sum(sigma(el, F{k}.cluster))/N;
                % temp(k) = temp(k) + sum(sigma(el, F{k}.cluster));
                rew(k) = sum(sqrt(temp));
            end
        end
        [scores(t),id] = max(rew);
        el = candidate(id);
        L = [L; el];
        F{id}.L = [F{id}.L; el];
        F{id}.fac = sum(max(sigma(F{id}.L,F{id}.cluster)));
        F{id}.V(find(F{id}.V==el)) = [];
        Reward(id) = Reward(id) + mean(sigma(el, F{id}.cluster));
        bestId = id; candidate(id) = 0;

        % rew = zeros(Group, 1);
        % parfor k = 1:Group
        %     if candidate(k) ~= 0
        %         temp = Number;
        %         temp(k) = temp(k) + 1;
        %         rew(k) = sum(sqrt(temp));
        %     end
        % end
        % [scores(t),id] = max(rew);
        % el = candidate(id);
        % L = [L; el];
        % F{id}.L = [F{id}.L; el];
        % F{k}.fac = sum(max(sigma(F{id}.L,F{k}.cluster)));
        % F{id}.V(find(F{id}.V==el)) = [];
        % Number(id) = Number(id) + 1;
        % bestId = id; candidate(id) = 0;
    end
    % n = [];
    % for k=1:Group
    %     n = [n length(F{k}.L)/N];
    % end
    % n
end

function [L,scores] = locfac_submodular(sigma, cluster, Group, B, L)
    N = length(cluster);
    scores = zeros(B,1);  % keep track of statistics
    weight = zeros(B,1);

    F = cell(Group, 1);
    for k = 1:Group
        idx = find(cluster==k);
        F{k}.cluster = idx;
        F{k}.fac = 0;
        F{k}.V = idx;
        F{k}.L = [];
        F{k}.N = length(idx);
        weight = F{k}.N;
    end

    bestId = 0;
    candidate = zeros(Group,1);
    for t = 1:B
        % find facility location candidates for each cluster
        if bestId == 0
            parfor k = 1:Group
                if length(F{k}.V) ~= 0
                    facility = zeros(1, length(F{k}.V)); % facility value for each left element in V
                    for i = 1:length(F{k}.V)
                        el = F{k}.V(i);
                        temp = [F{k}.L; el];
                        facility(i) = sum(max(sigma(temp,F{k}.cluster))) - F{k}.fac;
                    end
                    [~,id] = max(facility);
                    candidate(k) = F{k}.V(id);
                end
            end
        else
            if length(F{bestId}.V) ~= 0
                facility = zeros(1, length(F{bestId}.V)); % facility value for each left element in V
                for i = 1:length(F{bestId}.V)
                    el = F{bestId}.V(i);
                    temp = [F{bestId}.L; el];
                    facility(i) = sum(max(sigma(temp,F{bestId}.cluster)));
                end
                [~,id] = max(facility);
                candidate(bestId) = F{bestId}.V(id);
            end
        end

        index = randperm(Group);
        xx = 1;
        id = index(xx);
        el = candidate(id);
        while el == 0
            xx = xx+1;
            id = index(xx);
            el = candidate(id);
        end
        L = [L; el];
        F{id}.L = [F{id}.L; el];
        F{id}.fac = sum(max(sigma(F{id}.L,F{id}.cluster)));
        F{id}.V(find(F{id}.V==el)) = [];
        bestId = id; candidate(id) = 0;
    end
end

function [L,scores] = div_submodular(sigma, cluster, Group, B, L)
    N = length(cluster);
    scores = zeros(B,1);  % keep track of statistics
    weight = zeros(B,1);

    F = cell(Group, 1);
    for k = 1:Group
        idx = find(cluster==k);
        F{k}.cluster = idx;
        F{k}.fac = 0;
        F{k}.V = idx;
        F{k}.L = [];
        F{k}.N = length(idx);
        weight = F{k}.N;
    end
    Reward = zeros(Group,1);
    Number = ones(Group,1);

    bestId = 0;
    candidate = zeros(Group,1);
    for t = 1:B
        % find facility location candidates for each cluster
        if bestId == 0
            parfor k = 1:Group
                if length(F{k}.V) ~= 0
                    index = randperm(length(F{k}.V));
                    id = index(1);
                    candidate(k) = F{k}.V(id);
                end
            end
        else
            if length(F{bestId}.V) ~= 0
                index = randperm(length(F{bestId}.V));
                id = index(1);
                candidate(bestId) = F{bestId}.V(id);
            end
        end

        % find optimal element that yields maximal diversity value
        rew = zeros(Group, 1);
        parfor k = 1:Group
            if candidate(k) ~= 0
                el = candidate(k);
                temp = Reward;
                temp(k) = temp(k) + sum(sigma(el, F{k}.cluster))/N;
                % temp(k) = temp(k) + sum(sigma(el, F{k}.cluster));
                rew(k) = sum(sqrt(temp));
            end
        end
        [scores(t),id] = max(rew);
        el = candidate(id);
        L = [L; el];
        F{id}.L = [F{id}.L; el];
        F{id}.V(find(F{id}.V==el)) = [];
        Reward(id) = Reward(id) + mean(sigma(el, F{id}.cluster));
        bestId = id; candidate(id) = 0;
    end
end