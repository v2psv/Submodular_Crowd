function val = subsref(F,s)
% Implement a special subscripted assignment
   switch s.type
   case '()'
      A = s.subs{:};
      [tmp,val] = init(F,A);
   otherwise
      error('Invalid access')
   end
end
