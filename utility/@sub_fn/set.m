function a = set(a,varargin)
    propertyArgIn = varargin;
    while length(propertyArgIn) >= 2,
       prop = propertyArgIn{1};
       val = propertyArgIn{2};
       propertyArgIn = propertyArgIn(3:end);
       switch prop
       case 'current_val'
          a.current_val = val;
       case 'current_set'
          a.current_set = val;
       otherwise
          error('sub_fn properties: current_val, current_set')
       end
    end
end