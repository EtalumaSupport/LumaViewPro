%% Z-Stack — Capture images at multiple Z heights
%
% Captures a stack of BF images at regular Z intervals for 3D analysis.

scope = lvp_connect();

scope.set_objective('20x Oly');
scope.set_exposure(5);
scope.set_gain(1.0);

scope.move('X', 60000, true);
scope.move('Y', 40000, true);

%% Z-stack parameters
z_center = 5000;    % um — approximate focus
z_range  = 100;     % um — total range (above + below center)
z_step   = 5;       % um — step size

z_start = z_center - z_range/2;
z_end   = z_center + z_range/2;
z_positions = z_start:z_step:z_end;
n_slices = length(z_positions);

fprintf('Z-stack: %d slices, %.0f to %.0f um, step %.0f um\n', ...
    n_slices, z_start, z_end, z_step);

%% Capture stack
stack = zeros([1900, 1900, n_slices], 'uint8');  % pre-allocate

scope.led_on('BF', 100);

for i = 1:n_slices
    scope.move('Z', z_positions(i), true);
    stack(:,:,i) = scope.capture();

    if mod(i, 10) == 0
        fprintf('  Slice %d/%d (Z=%.0f um)\n', i, n_slices, z_positions(i));
    end
end

scope.leds_off();

%% Find best focus (max variance)
focus_scores = zeros(1, n_slices);
for i = 1:n_slices
    focus_scores(i) = var(double(stack(:,:,i)), 0, 'all');
end

[~, best_idx] = max(focus_scores);
fprintf('Best focus at Z=%.0f um (slice %d)\n', z_positions(best_idx), best_idx);

%% Display
figure;
subplot(1,2,1);
plot(z_positions, focus_scores, '-o');
xlabel('Z position (um)');
ylabel('Focus score (variance)');
title('Focus Curve');
xline(z_positions(best_idx), 'r--', sprintf('Best: %.0f um', z_positions(best_idx)));

subplot(1,2,2);
imshow(stack(:,:,best_idx));
title(sprintf('Best Focus — Z=%.0f um', z_positions(best_idx)));
