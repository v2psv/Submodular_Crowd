function [F,v] = init(F,sset)
    sset = sub_unique_fast(sset);
    if ~isequal(sset,get(F,'current_set'))
        if isempty(sset)
            F.rho = zeros(1,size(F.sigma,1));
            F.rew = zeros(1,max(F.cluster));
        else
            F.rho = max(F.sigma(sset,:));
            c = F.cluster(sset);
            for i=1:length(sset)
                F.rew(c(i)) = F.rew(c(i)) + mean(F.sigma(sset(i),:));
            end
        end
        F.facility = sum(F.rho);
        F.diversity = sum(sqrt(F.rew));
        F = set(F,'current_val',F.lambda*F.facility+(1-F.lambda)*F.diversity,'current_set',sset);
    end
    v = get(F,'current_val');
end
