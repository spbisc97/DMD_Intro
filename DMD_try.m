
% Clear workspace and close figures
clear; close all; clc;

Position=[10 1000 800 500];

set(groot,'defaultfigureposition',Position)
plotfolder='Plots/DMD_try/';


%% Generate Synthetic Data
% Define system dynamics: x_{k+1} = A * x_k
A = [0.9 0.1; -0.2 0.8];  % Example 2x2 system matrix

% Initial state
x0 = [1; 0];

% Number of time steps
numSteps = 50;

% Preallocate state matrix
n = length(x0);
X = zeros(n, numSteps);

% Generate data
X(:,1) = x0;
for k = 2:numSteps
    X(:,k) = A * X(:,k-1);
end

% Add small noise to simulate measurement noise
noise_level = 0.01;
X_noisy = X + noise_level * randn(size(X));

%% DMD Algorithm
% Split data into snapshots X1 and X2
X1 = X_noisy(:, 1:end-1);
X2 = X_noisy(:, 2:end);

% Perform Singular Value Decomposition (SVD) on X1
[U, Sigma, V] = svd(X1, 'econ');

% Truncate to rank-r (r can be chosen based on singular values)
r = 2;  % For this simple system, use full rank
U_r = U(:, 1:r);
Sigma_r = Sigma(1:r, 1:r);
V_r = V(:, 1:r);

% Compute low-rank approximation of A
A_tilde = U_r' * X2 * V_r / Sigma_r;

% Compute eigenvalues and eigenvectors of A_tilde
[W, D] = eig(A_tilde);

% DMD modes
Phi = X2 * V_r / Sigma_r * W;

% Compute eigenvalues (diagonal elements of D)
lambda = diag(D);

% Compute DMD mode amplitudes (initial conditions)
b = Phi \ X_noisy(:,1);

%% Reconstruct Dynamics using DMD Modes
% Time vector
t = 0:numSteps-1;

% Initialize reconstruction matrix
X_dmd = zeros(n, numSteps);
diagLambda=diag(lambda);


for k = 1:numSteps
    % X_dmd(:,k) = Phi * ((lambda.^(k-1)) .* b);
    for j=1:r
        X_dmd(:,k) = X_dmd(:,k) + (b(j).*Phi(:,j).*(lambda(j).^(k-1)));
    end
end

%% Plot Results
figure;
subplot(2,2,1);
plot(t, X_noisy(1,:), 'b-', 'LineWidth', 2); hold on;
plot(t, X_dmd(1,:), 'r--', 'LineWidth', 2);hold on;
plot(t, X(1,:), 'g--', 'LineWidth', 1);
xlabel('Time Step');
ylabel('State x_1');
legend('Original + noise', 'DMD Reconstruction','Original');
title('Comparison of Original and DMD Reconstructed States');
grid on;

subplot(2,2,3);
plot(t, X_noisy(2,:), 'b-', 'LineWidth', 2); hold on;
plot(t, X_dmd(2,:), 'r--', 'LineWidth', 2);
plot(t, X(2,:), 'g--', 'LineWidth', 1);
xlabel('Time Step');
ylabel('State x_2');
legend('Original + noise', 'DMD Reconstruction','Original');
grid on;

subplot(2,2,[2,4]);
plot(eig(A),'ro','LineWidth', 2);hold on;
plot(lambda,'go','LineWidth', 2);hold on;
rectangle('Position',[-1,-1,2,2],'Curvature',[1 1])
legend('DMD Reconstruction','Original')
xlim([-1 1])
ylim([-1 1])
grid on;

%% Display Results
disp('Original System Matrix A:');
disp(A);

disp('Reconstructed System Matrix from DMD:');
A_dmd = Phi * D / Phi;  % Reconstruct A from DMD modes and eigenvalues
disp(real(A_dmd));

disp('Eigenvalues of Original A:');
disp(eig(A));

disp('Eigenvalues from DMD:');
disp(lambda);
