%% 4x4 System DMD Example

% Clear workspace
clear; close all; clc;
Position=[10 1000 800 1000];
set(groot,'defaultfigureposition',Position)
plotfolder='Plots/ReduceDMD/';
%% Start

% Parameters
n = 4;             % State dimension
numSteps = 50;     % Number of time steps

% Time vector
t = 0:numSteps-1;

% Define system matrix A with full coupling and two dominant modes
A = [0.9, 0.3, 0.1, 0.0;
     -0.2, 0.8, 0.1, 0.2;
     0.1, -0.3, 0.7, -0.1;
     0.0, 0.2, -0.1, 0.6];

% Initial condition
x0 = [1; -1; 0.5; 0.2];

% Generate data
X = zeros(n, numSteps);
X(:,1) = x0;
for k = 2:numSteps
    X(:,k) = A * X(:,k-1);
end

% Add small noise to simulate measurement noise
noise_lvl = 0.0;
X_noisy = X + noise_lvl * randn(size(X));

%% Initial Plot
figure(1);
for i = 1:n
    subplot(n,1,i);
    plot(t, X(i,:), 'k-.', 'LineWidth', 1); hold on;
    plot(t, real(X_noisy(i,:)), 'go-', 'LineWidth', 2);hold on;
    xlabel('Time Step');
    ylabel(['State x', num2str(i)]);
    legend('Original','Noisy');
    title(['State x', num2str(i)]);
end

% Save plot 
saveas(gcf,fullfile(plotfolder,'Original_Noisy.png'));

%% DMD Algorithm

% Prepare snapshot matrices
X1 = X_noisy(:, 1:end-1);
X2 = X_noisy(:, 2:end);

% Perform SVD on X1
[U, Sigma, V] = svd(X1, 'econ');

% Plot singular values
fig2=figure(2);
semilogy(diag(Sigma), 'ro', 'LineWidth', 2); hold
% plot(diag(Sigma)/sum(diag(Sigma)), 'ro', 'LineWidth', 2);


xlabel('Singular Value Index');
ylabel('Singular Value Prominence');
title('Singular Values of X1');
grid on;
fig2.Position=[10 1000 500 550];

saveas(gcf,fullfile(plotfolder,'SingularValues.png'));
% Choose rank r based on singular values
r = 2;  % Truncate to rank-2 approximation
U_r = U(:, 1:r);
Sigma_r = Sigma(1:r, 1:r);
V_r = V(:, 1:r);

% Compute low-rank approximation of A
A_tilde = U_r' * X2 * V_r / Sigma_r;

% Compute eigenvalues and eigenvectors
[W, D] = eig(A_tilde);
lambda = diag(D);
Phi = X2 * V_r / Sigma_r * W;

% Compute mode amplitudes
b = Phi \ X_noisy(:,1);

%% Reconstruct Dynamics Using DMD Modes



% Initialize reconstruction matrix
X_dmd = zeros(n, numSteps);

% Reconstruct the dynamics
for k = 1:numSteps
    X_dmd(:,k) = Phi * (lambda.^(k-1) .* b);
end

%% Visualization

% Plot original and reconstructed trajectories
figure(1);
for i = 1:n
    subplot(n,1,i);
    hold off;
    plot(t, X(i,:), 'k-.', 'LineWidth', 1); hold on;
    plot(t, real(X_noisy(i,:)), 'g.-', 'LineWidth', 1);hold on;
    plot(t, real(X_dmd(i,:)), 'r--', 'LineWidth', 1); hold on;


    xlabel('Time Step');
    ylabel(['State x', num2str(i)]);
    legend('Original','Noisy', 'DMD Reconstruction');
    title(['State x', num2str(i)]);
end

saveas(gcf,fullfile(plotfolder,'Original_Noisy_DMD.png'));

[~,Dfull]=eig(U' * X2 * V / Sigma);
lambdafull=diag(Dfull);

figure(3)
subplot(2,1,1)

semilogy(diag(Sigma), 'ro', 'LineWidth', 2);
xlabel('Singular Value Index');
ylabel('Singular Value Prominence');
title('Singular Values of X1');


subplot(2,1,2)
hold off;
plot(eig(A),'ko','LineWidth', 2);hold on;
plot(lambda,'ro','LineWidth', 1);hold on;
plot(lambdafull,'b.','LineWidth', 2);hold on;
rectangle('Position',[-1,-1,2,2],'Curvature',[1 1])
legend('Original','Reduced DMD Reconstruction','fullyDMD Reconstruction')
xlim([-1 1])
ylim([-1 1])
grid on;

saveas(gcf,fullfile(plotfolder,'SingularValues_Comparison.png'));



% Calculate reconstruction error
reconstruction_error = norm(X - real(X_dmd), 'fro') / norm(X, 'fro');
disp(['Reconstruction Error (Relative Frobenius Norm): ', num2str(reconstruction_error)]);
