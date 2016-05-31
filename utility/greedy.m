% function [sset,scores] = greedy(F,V,B,initial_sset)
% F: submodular function
% V: index set
% B: budget

% returns solution sset with C(sset)<=B
% scores(i) is the greedy score after element i was added

function [sset,scores] = greedy(F,V,B,initial_sset)
    sset = initial_sset;                % start with empty set or specified
    scores = zeros(B,1);                % keep track of statistics

    for i=1:B
        % fprintf('Iteration: %d\n',i);
        [F,curVal] = init(F,sset);

        improv = zeros(1,length(V));
        for x = 1:length(V)
            improv(x) = inc(F,sset,V(x)) - curVal;
        end
        [~,id] = max(improv);
        sset = [sset,V(id)]; V(id) = [];

        scores(i) = curVal;
    end
end