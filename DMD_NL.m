
% %%%%%%%%% stolen from nat kutz




% Combined DMD and Time-Delay DMD Example on Van der Pol Oscillator



% Clear workspace and close figures
clear; close all; clc;
Position=[10 1000 800 800];

set(groot,'defaultfigureposition',Position)
plotfolder='Plots/DMD_NL/';

%% Generate Synthetic Data from Van der Pol Oscillator
% Parameters
mu = 0.8;

% Time vector
dt = 0.01;
t = 0:dt:50;

% Initial condition
x0 = [2; 0];

% Generate data using ODE45
[~, X] = ode45(@(t, x) [x(2); mu*(1 - x(1)^2)*x(2) - x(1)], t, x0);
X = X';  % Transpose to match dimensions

% Add small noise
noise_level = 0;
X_noisy = X + noise_level * randn(size(X));

%% Standard DMD
% Define snapshots
X1 = X_noisy(:, 1:end-1);
X2 = X_noisy(:, 2:end);

% Perform SVD
[U, Sigma, V] = svd(X1, 'econ');

% Truncate to rank-r
r_std = 2;  % Adjust rank as needed
U_r = U(:, 1:r_std);
Sigma_r = Sigma(1:r_std, 1:r_std);
V_r = V(:, 1:r_std);

% Compute A_tilde
A_tilde = U_r' * X2 * V_r / Sigma_r;

% Eigen decomposition
[W, D] = eig(A_tilde);
Phi_std = X2 * V_r / Sigma_r * W;

% Compute eigenvalues
lambda = diag(D);
omega = log(lambda)/dt;

% Compute mode amplitudes
b = Phi_std \ X_noisy(:,1);

% Reconstruct Dynamics
time_dynamics = zeros(r_std, length(t));
for k = 1:length(t)
    time_dynamics(:,k) = b .* exp(omega * t(k));
end
X_dmd_std = real(Phi_std * time_dynamics);

%% DMD with Time-Delay Embedding
% Embedding dimension (number of delays)
d = 200;  % Adjust as needed
n = size(X_noisy, 1);  % Number of state variables

% Construct time-delay embedded snapshots
m = size(X_noisy, 2) - d;  % Number of delay-embedded snapshots

% Initialize matrices
Z = zeros(n * d, m);
Z_prime = zeros(n * d, m);

for i = 1:m
    % Construct Z (delay-embedded snapshots)
    Z(:, i) = reshape(X_noisy(:, i:i+d-1), n*d, 1);
    % Construct Z_prime (shifted delay-embedded snapshots)
    Z_prime(:, i) = reshape(X_noisy(:, i+1:i+d), n*d, 1);
end

% Perform SVD on Z
[U_z, Sigma_z, V_z] = svd(Z, 'econ');

% Truncate to rank-r
r_td = d*2;  % Adjust rank as needed
U_r_z = U_z(:, 1:r_td);
Sigma_r_z = Sigma_z(1:r_td, 1:r_td);
V_r_z = V_z(:, 1:r_td);

% Compute A_tilde
A_tilde_z = U_r_z' * Z_prime * V_r_z / Sigma_r_z;

% Eigen decomposition
[W_z, D_z] = eig(A_tilde_z);
Phi_td = U_r_z * W_z;

% Continuous-time eigenvalues
lambda_z = diag(D_z);
omega_z = log(lambda_z)/dt;

% Compute mode amplitudes
x1_z = Z(:,1);  % Initial delay-embedded state
b_z = Phi_td \ x1_z;

% Reconstruct Dynamics
time_dynamics_z = zeros(r_td, m);
for i = 1:m
    time_dynamics_z(:, i) = b_z .* exp(omega_z * (i-1) * dt);
end

Z_dmd = real(Phi_td * time_dynamics_z);

% Extract the original state variables from the reconstructed Z_dmd
X_dmd_td = zeros(n, m + d - 1);
overlap_count = zeros(1, m + d - 1);

for i = 1:m
    % Extract the reconstructed delay-embedded vector
    z_temp = Z_dmd(:, i);
    % Reshape back to [n x d] matrix
    z_matrix = reshape(z_temp, n, d);
    % Sum overlapping predictions
    X_dmd_td(:, i:i+d-1) = X_dmd_td(:, i:i+d-1) + z_matrix;
    % Keep track of the number of overlaps
    overlap_count(i:i+d-1) = overlap_count(i:i+d-1) + 1;
end

% Average overlapping predictions
for i = 1:n
    X_dmd_td(i, :) = X_dmd_td(i, :) ./ overlap_count;
end

% Adjust time vector for reconstructed data
t_td = t(1:length(X_dmd_td));

%% Plot Results
figure;

% Plot State x_1
subplot(2,1,1);
plot(t, X(1,:), 'k-', 'LineWidth', 1.5); hold on;
plot(t, X_dmd_std(1,:), 'b--', 'LineWidth', 1);
plot(t_td, X_dmd_td(1,:), 'r-.', 'LineWidth', 1);
xlabel('Time');
ylabel('State x_1');
legend('Original', 'Standard DMD', 'Time-Delay DMD');
title('Comparison of DMD Reconstructions (State x_1)');
grid on;

% Plot State x_2
subplot(2,1,2);
plot(t, X(2,:), 'k-', 'LineWidth', 1.5); hold on;
plot(t, X_dmd_std(2,:), 'b--', 'LineWidth', 1);
plot(t_td, X_dmd_td(2,:), 'r-.', 'LineWidth', 1);
xlabel('Time');
ylabel('State x_2');
legend('Original', 'Standard DMD', 'Time-Delay DMD');
title('Comparison of DMD Reconstructions (State x_2)');
grid on;

%% Compute Reconstruction Errors
% Truncate data to common time range for error computation
min_length = min([length(X_dmd_std), length(X_dmd_td), size(X,2)]);
X_original = X(:, 1:min_length);
X_dmd_std_trunc = X_dmd_std(:, 1:min_length);
X_dmd_td_trunc = X_dmd_td(:, 1:min_length);

% Compute errors
error_std = norm(X_original - X_dmd_std_trunc, 'fro') / norm(X_original, 'fro');
error_td = norm(X_original - X_dmd_td_trunc, 'fro') / norm(X_original, 'fro');

% Display Errors
fprintf('Relative Reconstruction Error (Standard DMD): %.4f\n', error_std);
fprintf('Relative Reconstruction Error (Time-Delay DMD): %.4f\n', error_td);

% Plot SVD sigma

figure(2)
subplot(2,1,1)
% i prefer log instead of sum 
% plot(diag(Sigma_z)/sum(diag(Sigma_z)),'ro')
semilogy(diag(Sigma_z),'ro')

subplot(2,1,2)
% plot(diag(Sigma)/sum(diag(Sigma)),'ro')
semilogy(diag(Sigma),'ro')
% expand bottom ylim as the first one
ylim([90 1.1*max(diag(Sigma))])
xlim([1 3])