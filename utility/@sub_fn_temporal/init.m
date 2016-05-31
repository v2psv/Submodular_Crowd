function [F,v] = init(F,sset)
    sset = sub_unique_fast(sset);
    if ~isequal(sset,get(F,'current_set'))
        if isempty(sset)
            % F.tmp_dist = F.V.^2;
            F.tmp_dist = ones(size(F.V)) * max(F.V)^2;
        else
            F.tmp_dist = min(F.sigma(sset,:));
        end
        F = set(F,'current_val',F.phantom - sum(F.tmp_dist),'current_set',sset);
    end
    v = get(F,'current_val');
end
