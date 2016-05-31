function new_val = dec(F,A,el)
    F = init(F,sfo_setdiff_fast(A, el));
    new_val = get(F,'current_val');
end