% Helper for quickly computing the unique set representation
%
% function C=sub_unique_fast(A)
% A: set (array) of positive integers
%
% Example: C = sub_unique_fast([1 3 3 7 2 3 2 8])

function C=sub_unique_fast(A)
    mx = max(A);
    vals = zeros(1,mx);
    vals(A)=1;
    C = find(vals);
end