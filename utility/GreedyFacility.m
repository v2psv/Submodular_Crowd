function [L,scores] = GreedyFacility(sigma, B, init)
    % sigma: affinity matrix
    % cluster: clusters
    % Group: number of clusters
    % B: budget
    % init: initial set
    if nargin < 5
        L = [];
    else
        L = init;
    end
    scores = zeros(B,1);

    N = size(sigma, 1);
    V = [1:N];
    Fac = 0;

    for t=1:B
        facility = zeros(1, length(V));
        parfor i=1:length(V)
            el = V(i);
            temp = [L el];
            facility(i) = sum(max(sigma(temp,:))) - Fac;
        end
        [~,id] = max(facility);
        L = [L V(id)];
        V(id) = [];
        Fac = sum(max(sigma(L,:)));
    end
end