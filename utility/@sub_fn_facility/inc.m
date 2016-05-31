function newScore = inc(F,A,el)
    A = sub_unique_fast(A);
    F = init(F,A);

    if sum(A==el)>0
        newScore = get(F,'current_val');
        return;
    end

    if isempty(A)
        H = sum(F.sigma(el,:));
    else
        H = sum(max(F.sigma(el,:),F.rho));
    end

    newScore = H;
end