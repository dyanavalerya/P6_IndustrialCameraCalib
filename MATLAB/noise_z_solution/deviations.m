function distance_to_plane = deviations(ptCloud, Pb, X)
    % Calculate those vectors that go from the measurement pts perpendicular to
    % the plane
    vector_perp_plane = [ptCloud(:, 1) ptCloud(:, 2) ptCloud(:, 3)] - Pb';
    
    
    % the normal to the plane n = [A, B, C]. A and B were computed, C was set
    % to 1.
    n = [X(1), X(2), 1];
    
    % get the size of the data set
    N = length(ptCloud);
    % initialize the variable that stores the deviations
    distance_to_plane = zeros(N, 1);
    for i=1:N
        vpp = vector_perp_plane(i, :);

        % dot(n, vpp) gets the projection of vpp onto the normal
        % if the normal is in the opposite direction of vpp, then
        % the sign of this result will be negative, essentially telling 
        % that the measurement point is below the surface
        % norm of vpp is the euclidean distance, aka deviation
        distance_to_plane(i) = sign(dot(n, vpp))*norm(vpp, 2);
    end
end