function newScore = inc(F,A,el)
    A = sub_unique_fast(A);
    F = init(F,A);

    if sum(A==el)>0
        newScore = get(F,'current_val');
        return;
    end

    if isempty(A)
        H = sum(F.sigma(el,:));
        newScore = sqrt(mean(F.sigma(el,:)));
    else
        c = F.cluster(el);
        tmp_rew = F.rew;
        tmp_rew(c) = tmp_rew(c) + mean(F.sigma(el,:));
        newScore = sum(sqrt(tmp_rew));
    end
end
