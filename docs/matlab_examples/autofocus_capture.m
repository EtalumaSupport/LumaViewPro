%% Autofocus + Capture — Focus then acquire
%
% Runs autofocus at the current position, then captures with the
% best focus Z position.

scope = lvp_connect();

scope.set_objective('10x Oly');

scope.move('X', 60000, true);
scope.move('Y', 40000, true);
scope.move('Z', 5000, true);    % approximate focus

%% Run autofocus via Z-stack + focus scoring
% The REST API will provide a dedicated /autofocus endpoint.
% Until then, this demonstrates the manual approach.

z_center = scope.get_position('Z');
z_range  = 200;     % um search range
z_step   = 10;      % um coarse step

z_positions = (z_center - z_range/2):z_step:(z_center + z_range/2);
n_slices = length(z_positions);

%% Coarse pass — find approximate focus
scores = zeros(1, n_slices);

scope.led_on('BF', 100);

for i = 1:n_slices
    scope.move('Z', z_positions(i), true);
    img = scope.capture();
    scores(i) = var(double(img), 0, 'all');
end

[~, best_idx] = max(scores);
z_coarse = z_positions(best_idx);
fprintf('Coarse focus: Z=%.0f um\n', z_coarse);

%% Fine pass — refine around coarse result
z_fine_step = 2;    % um
z_fine_range = z_step * 2;
z_fine = (z_coarse - z_fine_range/2):z_fine_step:(z_coarse + z_fine_range/2);
n_fine = length(z_fine);
scores_fine = zeros(1, n_fine);

for i = 1:n_fine
    scope.move('Z', z_fine(i), true);
    img = scope.capture();
    scores_fine(i) = var(double(img), 0, 'all');
end

[~, best_fine_idx] = max(scores_fine);
z_best = z_fine(best_fine_idx);
fprintf('Fine focus: Z=%.1f um\n', z_best);

%% Move to best focus and capture final image
scope.move('Z', z_best, true);
img_focused = scope.capture();
scope.leds_off();

%% Display
figure;
subplot(1,3,1);
plot(z_positions, scores, '-o');
hold on;
xline(z_coarse, 'r--');
xlabel('Z (um)'); ylabel('Focus score');
title('Coarse Pass');

subplot(1,3,2);
plot(z_fine, scores_fine, '-o');
hold on;
xline(z_best, 'r--');
xlabel('Z (um)'); ylabel('Focus score');
title('Fine Pass');

subplot(1,3,3);
imshow(img_focused);
title(sprintf('Focused at Z=%.1f um', z_best));
