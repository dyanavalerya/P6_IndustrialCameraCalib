function Pb = projection_vector(ptCloud, X)
    % Methof 1 for getting the projection matrix

    % Pick two points on the plane that are linearly independent (not parallel)
    % pt = [x, y, z = -A*x - B*y]
    % These points span the plane
    % I avoid using the D constant because it will make the plane be affine
    % instead of linear
    % difference between affine and linear is that affine has an added constant
    pt1 = [1; 7; -X(1)*1-X(2)*7];
    pt2 = [4; -3; -X(1)*4-X(2)*(-3)];
    A2 = [pt1, pt2];
    P = A2/(A2'*A2)*A2';

    % cond_number2 = cond(A2)
    
    % Method 2
    plane_func = @(x,y)(-X(1)*x - X(2)*y - X(3));
    %n = [X(1); X(2); 1]; % Normal vector to plane
    %n = n/norm(n); % Normalized
    %P = eye(3) - n*n'; % Projection matrix - correct method as well
    
    % Create the new origin point
    % Write z as a function of x and y=0 and let c=1
    p0 = [0; 0; plane_func(0,0)];
    
    % Shift the plane by -d along the z-axis from the origin
    % Need to do that to bring the plane to the origin, otherwhise
    % cannot project
    shifted_points = ptCloud'-p0;

    % Project points into the new frame and transform back to original
    % frame
    Pb = P*shifted_points + p0;
end