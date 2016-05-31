function newScore = inc(F,A,el)
    A = sub_unique_fast(A);
    F = init(F,A);

    if sum(A==el)>0
        newScore = get(F,'current_val');
        return;
    end

    if isempty(A)
        H = F.phantom - sum(F.sigma(el,:));
    else
        H = F.phantom - sum(min(F.sigma(el,:),F.tmp_dist));
    end

    newScore = H;
end