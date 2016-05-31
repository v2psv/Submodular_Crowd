function F = sub_fn_mix(sigma, V, Group, cluster)
    F = sub_fn_mix(sigma, Ntrain, option.Group, cluster);
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