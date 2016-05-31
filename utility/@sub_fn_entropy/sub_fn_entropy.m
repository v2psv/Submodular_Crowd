function F = sub_fn_entropy(sigma,V)
    F.sigma = sigma;
    F.V = V;

    F.indsA = [];
    F.cholA = [];

    F = class(F,'sub_fn_entropy',sub_fn);
    F = set(F,'current_set',-1);
end