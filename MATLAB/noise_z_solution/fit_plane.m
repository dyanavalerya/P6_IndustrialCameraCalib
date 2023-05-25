function [X, A] = fit_plane(ptCloud)
    b = -ptCloud(:,3);
    A = zeros(length(b), 3);
    for i = 1:length(b)
        A(i, 1) = ptCloud(i, 1);
        A(i, 2) = ptCloud(i, 2);
        A(i, 3) = 1;
    end
    
    % condition_number = cond(A)
    
    % Calculate the A, B, D coefficients
    %X = least_squares(A, b);
    X = A\b;
end