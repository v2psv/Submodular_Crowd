function [F,v] = init(F,sset)
    sset = sub_unique_fast(sset);
    if ~isequal(sset,get(F,'current_set'))
        if isempty(sset)
            F.rho = zeros(1,size(F.sigma,1));
        else
            F.rho = max(F.sigma(sset,:));
        end
        F = set(F,'current_val',sum(F.rho),'current_set',sset);
    end
    v = get(F,'current_val');
end
