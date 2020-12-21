function [pc, eigenvalues] = mypca(A)
    %first step is to centre the data
    A = A-mean(A);

    %second step is to calculate the variances and covariances among every pair
    %of the p variables
    covariance = cov(A);

    %compute the eigenvalues and vectors
    [V,D] = eig(covariance);
    eigenvalues = diag(D);
    pc = V;
    
    [~,srtidx] = sort(eigenvalues,'descend');
    eigenvalues = eigenvalues(srtidx);
    pc = pc(:,srtidx);
    
end