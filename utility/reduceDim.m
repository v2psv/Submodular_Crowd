function TestTransformed = reduceDim(X, Y, method, dim)
    [TrainTransformed, mapping] = compute_mapping(X, method, dim);
    TrainTransformed = [TrainTransformed ones(size(TrainTransformed,1), 1)];
    TestTransformed = out_of_sample(X, mapping);
    TestTransformed = [TestTransformed ones(size(TestTransformed,1), 1)];
end