function [F,v] = init(F,sset)
    sset = sub_unique_fast(sset);
    if ~isequal(sset,get(F,'current_set'))
        if isempty(sset)
            F.rew = zeros(1,max(F.cluster));
        else
            c = F.cluster(sset);
            for i=1:length(sset)
                F.rew(c(i)) = F.rew(c(i)) + mean(F.sigma(sset(i),:));
            end
        end
        F = set(F,'current_val',sum(sqrt(F.rew)),'current_set',sset);
    end
    v = get(F,'current_val');
end
