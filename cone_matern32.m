% =========================================================
% PART 1: Cone Interpolation with Gaussian Kernel
% =========================================================

clear; clc; close all;
rng(42);

N  = 40;
ep = 3;

% Generate random points inside the unit disk
theta = 2 * pi * rand(N, 1);
r     = sqrt(rand(N, 1));
x     = r .* cos(theta);
y     = r .* sin(theta);

% Evaluate the cone function
f = 1 - sqrt(x.^2 + y.^2);

% Build Matern 3/2 kernel matrix
A = zeros(N, N);
for i = 1:N
    for j = 1:N
        d2     = (x(i) - x(j))^2 + (y(i) - y(j))^2;
        d = sqrt(d2);
        A(i,j) = (1 + ep*d) * exp(-ep*d);
    end
end

% Solve for coefficients
c = A \ f;

% Evaluate on grid
[xg, yg] = meshgrid(linspace(-1, 1, 80));
zg = nan(size(xg));

for k = 1:numel(xg)
    if xg(k)^2 + yg(k)^2 <= 1
        total = 0;
        for j = 1:N
            d2    = (xg(k) - x(j))^2 + (yg(k) - y(j))^2;
            d = sqrt(d2);
            total = total + c(j) * (1 + ep*d) * exp(-ep*d);
        end
        zg(k) = total;
    end
end

% Figure 1 - Cone surface
figure(1);
surf(xg, yg, zg, 'EdgeColor', 'none');
hold on;
scatter3(x, y, f, 40, 'r', 'filled');
title('Cone Interpolation with Matern 3/2 RBF');
xlabel('x'); ylabel('y'); zlabel('z');
view(3); colorbar;
hold off;

% =========================================================
% PART 2: Double Descent - Cone + Gaussian
% =========================================================

% Generate test points
Nt      = 200;
theta_t = 2 * pi * rand(Nt, 1);
r_t     = sqrt(rand(Nt, 1));
xt      = r_t .* cos(theta_t);
yt      = r_t .* sin(theta_t);
ft      = 1 - sqrt(xt.^2 + yt.^2);


% Generate well-distributed nested centers
M_max = 200;

% Create candidate points inside the unit disk
rng(42);
num_candidates = 5000;
theta_c = 2*pi*rand(num_candidates,1);
r_c = sqrt(rand(num_candidates,1));

candidates = [r_c.*cos(theta_c), r_c.*sin(theta_c)];

% Select centers using farthest-point sampling
centers = zeros(M_max,2);
centers(1,:) = [0,0];

min_dist2 = sum((candidates-centers(1,:)).^2,2);

for j = 2:M_max
    [~,idx] = max(min_dist2);
    centers(j,:) = candidates(idx,:);

    new_dist2 = sum((candidates-centers(j,:)).^2,2);
    min_dist2 = min(min_dist2,new_dist2);
end

xc_all = centers(:,1);
yc_all = centers(:,2);


% Loop over M up to 200

M_list = unique([3:3:200, N, 200]);
err_train_fix  = zeros(size(M_list));
err_test_fix   = zeros(size(M_list));
err_train_adap = zeros(size(M_list));
err_test_adap  = zeros(size(M_list));

for idx = 1:length(M_list)
    M = M_list(idx);

    % Nested: always keep previous centers
    xc = xc_all(1:M);
    yc = yc_all(1:M);

    % Fixed and adaptive shape parameter
    ep_fix  = 3;
    ep_adap = sqrt(M) / 2;

    % --- Fixed ep: training matrix ---
    B = zeros(N, M);
    for i = 1:N
        for j = 1:M
            d2     = (x(i) - xc(j))^2 + (y(i) - yc(j))^2;
            d = sqrt(d2);
            B(i,j) = (1 + ep_fix*d) * exp(-ep_fix*d);
        end
    end
    coef               = pinv(B) * f;
    f_approx           = B * coef;
    err_train_fix(idx) = norm(f - f_approx) / norm(f);

    % Fixed ep: test matrix
    Bt = zeros(Nt, M);
    for i = 1:Nt
        for j = 1:M
            d2      = (xt(i) - xc(j))^2 + (yt(i) - yc(j))^2;
            d = sqrt(d2);
            Bt(i,j) = (1 + ep_fix*d) * exp(-ep_fix*d);
        end
    end
    ft_approx         = Bt * coef;
    err_test_fix(idx) = norm(ft - ft_approx) / norm(ft);

    % --- Adaptive ep: training matrix ---
    B2 = zeros(N, M);
    for i = 1:N
        for j = 1:M
            d2      = (x(i) - xc(j))^2 + (y(i) - yc(j))^2;
            d = sqrt(d2);
            B2(i,j) = (1 + ep_adap*d) * exp(-ep_adap*d);
        end
    end
    coef2               = pinv(B2) * f;
    f_approx2           = B2 * coef2;
    err_train_adap(idx) = norm(f - f_approx2) / norm(f);

    % Adaptive ep: test matrix
    Bt2 = zeros(Nt, M);
    for i = 1:Nt
        for j = 1:M
            d2       = (xt(i) - xc(j))^2 + (yt(i) - yc(j))^2;
            d = sqrt(d2);
            Bt2(i,j) = (1 + ep_adap*d) * exp(-ep_adap*d);
        end
    end
    ft_approx2         = Bt2 * coef2;
    err_test_adap(idx) = norm(ft - ft_approx2) / norm(ft);

end

% Figure 2 - Fixed ep double descent
figure(2);
semilogy(M_list, err_train_fix, 'b-', 'LineWidth', 2);
hold on;
semilogy(M_list, err_test_fix, 'r-', 'LineWidth', 2);
xline(N, '--k', 'LineWidth', 1.5);
text(N + 1, max(err_test_fix) * 0.5, 'M = N', 'FontSize', 10);
xlabel('Number of Kernel Centers M');
ylabel('Relative Error (log scale)');
title('Cone + Matern 3/2: Fixed ep = 3, Nested Centers');
legend('Training Error', 'Test Error', 'Location', 'northeast');
grid on;
hold off;

% Figure 3 - Adaptive ep double descent
figure(3);
semilogy(M_list, err_train_adap, 'b-', 'LineWidth', 2);
hold on;
semilogy(M_list, err_test_adap, 'r-', 'LineWidth', 2);
xline(N, '--k', 'LineWidth', 1.5);
text(N + 1, max(err_test_adap) * 0.5, 'M = N', 'FontSize', 10);
xlabel('Number of Kernel Centers M');
ylabel('Relative Error (log scale)');
title('Cone + Matern 3/2: Adaptive ep = sqrt(M)/2, Nested Centers');
legend('Training Error', 'Test Error', 'Location', 'northeast');
grid on;
hold off;

% Figure 4 - Distribution of nested kernel centers
figure(4);

scatter(xc_all(1:40), yc_all(1:40), 40, 'b', 'filled');
hold on;

t = linspace(0, 2*pi, 300);
plot(cos(t), sin(t), 'k-', 'LineWidth', 1.5);

axis equal;
xlim([-1.1 1.1]);
ylim([-1.1 1.1]);
grid on;

xlabel('x');
ylabel('y');
title('Distribution of First 40 Nested Kernel Centers');

hold off;
