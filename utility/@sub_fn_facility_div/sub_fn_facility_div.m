function F = sub_fn_facility_div(sigma, V, lambda, cluster)
    F.sigma = sigma;
    F.V = V;

    F.rho = [];
    F.rew = [];
    F.lambda = lambda;
    F.cluster = cluster;

    F.facility = 0;
    F.diversity = 0;
    F = class(F,'sub_fn_facility_div',sub_fn);
    F = set(F,'current_set',-1);
end