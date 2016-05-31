function F = sub_fn_temporal(sigma,V)
    F.sigma = sigma;
    F.V = V;

    F.tmp_dist = [];
    % F.phantom = sum(sigma(1,:));
    % F.phantom = sum(F.V.^2);
    F.phantom = length(F.V) * max(F.V)^2;

    F = class(F,'sub_fn_temporal',sub_fn);
    F = set(F,'current_set',-1);
end