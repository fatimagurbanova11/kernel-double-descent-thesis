
% Compare Gaussian and Matern 3/2 kernels
clear; clc; close all;

G = load('gaussian_results.mat');
T = load('matern32_results.mat');

% Check that both experiments used the same model sizes
assert(isequal(G.M_list, T.M_list), ...
    'The two experiments use different M values.');

% Fixed shape parameter comparison
figure(1);
semilogy(G.M_list, G.err_test_fix, 'b-', 'LineWidth', 2);
hold on;
semilogy(T.M_list, T.err_test_fix, 'r-', 'LineWidth', 2);

xline(40, '--k', 'M = N', 'LineWidth', 1.5);

xlabel('Number of Kernel Centers M');
ylabel('Relative Test Error (log scale)');
title('Gaussian vs Matern 3/2: Cone Function, Fixed ep = 3');
legend('Gaussian', 'Matern 3/2', 'Location', 'best');
grid on;
hold off;
