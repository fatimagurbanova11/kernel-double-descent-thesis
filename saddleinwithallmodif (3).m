
% PART 1: Saddle Interpolation with Multiquadric Kernel

clear; clc; close all;
rng(42);

N  = 25;
ep = 3;

theta = 2 * pi * rand(N, 1);
r     = sqrt(rand(N, 1));
x     = r .* cos(theta);
y     = r .* sin(theta);

f = x.^2 - y.^2;


A = zeros(N, N);
for i = 1:N
    for j = 1:N
        d2     = (x(i) - x(j))^2 + (y(i) - y(j))^2;
        A(i,j) = sqrt(1 + ep^2 * d2);
    end
end


c = A \ f;


[xg, yg] = meshgrid(linspace(-1, 1, 80));
zg = nan(size(xg));

for k = 1:numel(xg)
    if xg(k)^2 + yg(k)^2 <= 1
        total = 0;
        for j = 1:N
            d2    = (xg(k) - x(j))^2 + (yg(k) - y(j))^2;
            total = total + c(j) * sqrt(1 + ep^2 * d2);
        end
        zg(k) = total;
    end
end


figure(1);
surf(xg, yg, zg, 'EdgeColor', 'none');
hold on;
scatter3(x, y, f, 40, 'r', 'filled');
title('Saddle Interpolation with Multiquadric RBF');
xlabel('x'); ylabel('y'); zlabel('z');
view(3); colorbar;
hold off;


% PART 2: Double Descent - Saddle + Multiquadric



Nt      = 200;
theta_t = 2 * pi * rand(Nt, 1);
r_t     = sqrt(rand(Nt, 1));
xt      = r_t .* cos(theta_t);
yt      = r_t .* sin(theta_t);
ft      = xt.^2 - yt.^2;


% Generate well-distributed nested kernel centers
M_max = 200;

rng(42);
num_candidates = 5000;
theta_c = 2*pi*rand(num_candidates,1);
r_c = sqrt(rand(num_candidates,1));

candidates = [r_c.*cos(theta_c), r_c.*sin(theta_c)];

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




M_list = unique([3:3:200, N, 200]);
err_train_fix  = zeros(size(M_list));
err_test_fix   = zeros(size(M_list));
err_train_adap = zeros(size(M_list));
err_test_adap  = zeros(size(M_list));

for idx = 1:length(M_list)
    M = M_list(idx);

    xc = xc_all(1:M);
    yc = yc_all(1:M);

    ep_fix  = 3;


    ep_adap = sqrt(M) / 2;

    B = zeros(N, M);
    for i = 1:N
        for j = 1:M
            d2     = (x(i) - xc(j))^2 + (y(i) - yc(j))^2;
            B(i,j) = sqrt(1 + ep_fix^2 * d2);
        end
    end
    coef               = pinv(B) * f;
    f_approx           = B * coef;
    err_train_fix(idx) = norm(f - f_approx) / norm(f);

    Bt = zeros(Nt, M);
    for i = 1:Nt
        for j = 1:M
            d2      = (xt(i) - xc(j))^2 + (yt(i) - yc(j))^2;
            Bt(i,j) = sqrt(1 + ep_fix^2 * d2);
        end
    end
    ft_approx         = Bt * coef;
    err_test_fix(idx) = norm(ft - ft_approx) / norm(ft);


    B2 = zeros(N, M);
    for i = 1:N
        for j = 1:M
            d2      = (x(i) - xc(j))^2 + (y(i) - yc(j))^2;
            B2(i,j) = sqrt(1 + ep_adap^2 * d2);
        end
    end
    coef2               = pinv(B2) * f;
    f_approx2           = B2 * coef2;
    err_train_adap(idx) = norm(f - f_approx2) / norm(f);


    Bt2 = zeros(Nt, M);
    for i = 1:Nt
        for j = 1:M
            d2       = (xt(i) - xc(j))^2 + (yt(i) - yc(j))^2;
            Bt2(i,j) = sqrt(1 + ep_adap^2 * d2);
        end
    end
    ft_approx2         = Bt2 * coef2;
    err_test_adap(idx) = norm(ft - ft_approx2) / norm(ft);

end


figure(2);
semilogy(M_list, err_train_fix, 'b-', 'LineWidth', 2);
hold on;
semilogy(M_list, err_test_fix, 'r-', 'LineWidth', 2);
xline(N, '--k', 'LineWidth', 1.5);
text(N + 1, max(err_test_fix) * 0.5, 'M = N', 'FontSize', 10);
xlabel('Number of Kernel Centers M');
ylabel('Relative Error (log scale)');
title('Saddle + Multiquadric: Fixed ep = 3, Nested Centers');
legend('Training Error', 'Test Error', 'Location', 'northeast');
grid on;
hold off;

figure(3);
semilogy(M_list, err_train_adap, 'b-', 'LineWidth', 2);
hold on;
semilogy(M_list, err_test_adap, 'r-', 'LineWidth', 2);
xline(N, '--k', 'LineWidth', 1.5);
text(N + 1, max(err_test_adap) * 0.5, 'M = N', 'FontSize', 10);
xlabel('Number of Kernel Centers M');
ylabel('Relative Error (log scale)');
title('Saddle + Multiquadric: Adaptive ep = sqrt(M)/2, Nested Centers');
legend('Training Error', 'Test Error', 'Location', 'northeast');
grid on;
hold off;


% Figure 4 - Distribution of nested kernel centers
figure(4);
scatter(xc_all(1:25), yc_all(1:25), 40, 'b', 'filled');
hold on;

t = linspace(0, 2*pi, 300);
plot(cos(t), sin(t), 'k-', 'LineWidth', 1.5);

axis equal;
xlim([-1.1 1.1]);
ylim([-1.1 1.1]);
grid on;

xlabel('x');
ylabel('y');
title('Distribution of First 25 Nested Kernel Centers');
hold off;
