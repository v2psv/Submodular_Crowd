function [F,value] = init(F,sset)
    sset = sub_unique_fast(sset);
    N_sset = length(sset);
    if ~isequal(sset,get(F,'current_set'))
        F.cholA = chol(F.sigma(sset,sset)+(1e-6)*eye(N_sset));
        F.indsA = sset;

        if isempty(sset)
            value = 0;
        else
    %         value = 1/2*log2((2*pi*exp(1))^size(F.cholA,1)) + sum(log2(diag(F.cholA)));
            value = 1/2*N_sset*log2((2*pi*exp(1))) + sum(log2(diag(F.cholA)));
        end
        F = set(F,'current_val',value,'current_set',sset);
    else
        value = get(F,'current_val');
    end
end