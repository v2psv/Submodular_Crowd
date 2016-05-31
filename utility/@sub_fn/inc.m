function new_val = inc(F,A,el)
    F = init(F,[A, el]);
    new_val = get(F,'current_val');
end
