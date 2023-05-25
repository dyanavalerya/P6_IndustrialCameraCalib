%This script illustrates a multivariate Gaussian distribution and its
%marginal distributions

%This code is issued under the CC0 "license"
% I got it from https://en.wikipedia.org/wiki/Multivariate_normal_distribution
% and edited based on my needs

clear all
close all

% Get the data
mean_ = [];
variance_ = [];

datapath = 'cam1/final_attempt';
d=dir(fullfile(datapath,'*.csv'));
for i=1:numel(d)
  path = fullfile(datapath,d(i).name);
  if contains(path, 'variance') == 0
      Samples = load(path);
      [mean, variance] = data_analysis(Samples, path, datapath);
      mean_ = [mean_; mean];
      variance_ = [variance_; variance];
  end
end

% These are the dimensions of the machined steel plate
real_width = 825; % mm
real_height = 440; % mm 
left_edge = norm(mean_(1,:)-mean_(3,:));
right_edge = norm(mean_(2,:)-mean_(4,:));
up_edge = norm(mean_(3,:)-mean_(4,:));
down_edge = norm(mean_(1,:)-mean_(2,:));
err_le = real_height - left_edge; % mm
err_re = real_height - right_edge;
err_ue = real_width - up_edge;
err_de = real_width - down_edge;

% Table containing the errors of real steel plate dimensions and the
% measured dimensions
err_dimensions = table(err_le, err_re, err_ue, err_de, 'VariableNames', ...
    {'Left edge err', 'Right edge err', 'Up edge err', 'Down edge err'});
writetable(err_dimensions, "dimensions_err_cam"+datapath(4)+".csv")

T = table(mean_, variance_, 'VariableNames', { 'Mean', 'Variance'});
%writetable(T, "mean_variance_cam"+datapath(4)+".csv");

function [MeanVec, variance] = data_analysis(Samples, path, folderpath)
    % Get the minimum and maximum values for x and y axes
    min_x = min(Samples(:, 1));
    min_y = min(Samples(:, 2));
    max_x = max(Samples(:, 1));
    max_y = max(Samples(:, 2));
    
    %2-d Mean and covariance matrix
    %MeanVec = [0 0];
    %CovMatrix = [1 0.6; 0.6 2];
    CovMatrix = cov(Samples);
    MeanVec = mean(Samples);
    variance = var(Samples);
    
    %Define limits for plotting
    const = 3;
    X = min_x-const:0.05:max_x+const;
    Y = min_y-const:0.05:max_y+const;
    
    %Get the 1-d PDFs for the "walls"
    Z_x = normpdf(X,MeanVec(1), sqrt(CovMatrix(1,1)));
    Z_y = normpdf(Y,MeanVec(2), sqrt(CovMatrix(2,2)));
    
    %Get the 2-d samples for the "floor"
    %Samples = mvnrnd(MeanVec, CovMatrix, 100);
    
    
    %Get the sigma ellipses by transform a circle by the cholesky decomp
    L = chol(CovMatrix,'lower');
    t = linspace(0,2*pi,100); %Our ellipse will have 100 points on it
    C = [cos(t); sin(t)]; %A unit circle
    E1 = 1*L*C; E2 = 2*L*C; E3 = 3*L*C; %Get the 1,2, and 3-sigma ellipses
    
    figure; hold on; 
    %Plot the samples on the "floor"
    plot3(Samples(:,1),Samples(:,2),zeros(size(Samples,1),1),'k.','MarkerSize',7)
    xlabel('x coordinates');
    ylabel('y coordinates');
    %Plot the 1,2, and 3-sigma ellipses slightly above the floor
    %plot3(E1(1,:), E1(2,:), 1e-3+zeros(1,size(E1,2)),'Color','g','LineWidth',2);
    %plot3(E2(1,:), E2(2,:), 1e-3+zeros(1,size(E2,2)),'Color','g','LineWidth',2);
    plot3(E3(1,:)+MeanVec(1), E3(2,:)+MeanVec(2), 1e-3+zeros(1,size(E3,2)),'Color','g','LineWidth',2);
    
    %Plot the histograms on the walls from the data in the middle
    [n_x, xout] = hist(Samples(:,1),20);%Creates 20 bars
    n_x = n_x ./ ( sum(n_x) *(xout(2)-xout(1)));%Normalizes to be a pdf
    [~,~,~,x_Pos,x_Height] = makebars(xout,n_x);%Creates the bar points
    plot3(x_Pos, Y(end)*ones(size(x_Pos)),x_Height,'-k')
    
    %Now plot the other histograms on the wall
    [n_y, yout] = hist(Samples(:,2),20);
    n_y = n_y ./ ( sum(n_y) *(yout(2)-yout(1)));
    [~,~,~,y_Pos,y_Height] = makebars(yout,n_y);
    plot3(X(1)*ones(size(y_Pos)),y_Pos, y_Height,'-k')
    
    %Now plot the 1-d pdfs over the histograms
    plot3(X, ones(size(X))*Y(end), Z_x,'-b','LineWidth',2); 
    plot3(ones(size(Y))*X(1), Y, Z_y,'-r','LineWidth',2);
    legend('Corner coordinates', '3-sigma ellipse', '', '', 'Pdf of X', 'Pdf of Y');
    % Prepare title name based on file name
    data_name = erase(path, [folderpath, "/", ".csv"]);
    data_name = strrep(data_name, '_', ' ');
    % Switch information around to make it look better
    data_name = strcat("Camera ", data_name(end), ": ", erase(data_name, ["cam", data_name(end)]));
    title(data_name);
    
    %Make the figure look nice
    grid on; view(45,55);
    axis([X(1) X(end) Y(1) Y(end)])

    % Save the figure
    %saveas(gcf, erase(path, [folderpath, "/", ".csv"]), 'png')
end
