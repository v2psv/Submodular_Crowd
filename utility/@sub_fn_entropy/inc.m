function newScore = inc(F,A,el)
    A = sub_unique_fast(A);
    F = init(F,A);

    if sum(A==el)>0
        newScore = get(F,'current_val');
        return;
    end

    if (isempty(A))
        sigmaXgA = F.sigma(el,el);
    else
        sigmaXgA = F.sigma(el,el)-F.sigma(el,A)*(F.cholA\(F.cholA'\F.sigma(A,el)));
    end

    H = 1/2*log2(2*pi*exp(1)) + 1/2*sum(log2(sigmaXgA));

    newScore = get(F,'current_val') + H;
end