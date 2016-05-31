function F = sub_fn_facility(sigma,V)
    F.sigma = sigma;
    F.V = V;

    F.rho = [];

    F = class(F,'sub_fn_facility',sub_fn);
    F = set(F,'current_set',-1);
end