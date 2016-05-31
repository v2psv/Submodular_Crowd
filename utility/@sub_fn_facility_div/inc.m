function newScore = inc(F,A,el)
    A = sub_unique_fast(A);
    F = init(F,A);

    if sum(A==el)>0
        newScore = get(F,'current_val');
        return;
    end

    if isempty(A)
        H = sum(F.sigma(el,:));
        newScore = F.lambda*H + (1-F.lambda)*sqrt(mean(F.sigma(el,:)));
    else
        H = sum(max(F.sigma(el,:),F.rho));
        c = F.cluster(el);
        tmp_rew = F.rew;
        tmp_rew(c) = tmp_rew(c) + mean(F.sigma(el,:));
        newScore = F.lambda*H + (1-F.lambda)*sum(sqrt(tmp_rew));
    end
end
