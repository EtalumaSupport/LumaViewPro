%% Timelapse — Repeated captures at a fixed interval
%
% Captures BF images every N seconds for a total duration.
% Useful for monitoring cell growth, drug response, etc.

scope = lvp_connect();

scope.set_objective('10x Oly');
scope.set_exposure(10);
scope.set_gain(3.0);

scope.move('X', 60000, true);
scope.move('Y', 40000, true);
scope.move('Z', 5000, true);

%% Timelapse parameters
interval_s  = 60;     % seconds between captures
duration_hr = 2;      % total duration in hours
n_frames = floor(duration_hr * 3600 / interval_s);

fprintf('Timelapse: %d frames, every %ds, total %.1f hours\n', ...
    n_frames, interval_s, duration_hr);

%% Capture loop
timestamps = zeros(1, n_frames);
save_dir = 'C:\data\timelapse';
if ~exist(save_dir, 'dir'); mkdir(save_dir); end

scope.led_on('BF', 100);

for i = 1:n_frames
    t_start = tic;
    timestamps(i) = now;

    % Capture
    img = scope.capture();

    % Save with timestamp in filename
    fname = fullfile(save_dir, sprintf('frame_%04d_%s.tiff', i, ...
        datestr(timestamps(i), 'yyyymmdd_HHMMSS')));
    imwrite(img, fname);

    fprintf('Frame %d/%d saved: %s\n', i, n_frames, fname);

    % Wait for next interval
    elapsed = toc(t_start);
    pause_time = max(0, interval_s - elapsed);
    if i < n_frames
        pause(pause_time);
    end
end

scope.leds_off();

%% Plot intensity over time
mean_intensities = cellfun(@(f) mean(imread(f), 'all'), ...
    arrayfun(@(i) fullfile(save_dir, sprintf('frame_%04d_%s.tiff', i, ...
    datestr(timestamps(i), 'yyyymmdd_HHMMSS'))), 1:n_frames, 'UniformOutput', false));

figure;
t_minutes = (timestamps - timestamps(1)) * 24 * 60;
plot(t_minutes, mean_intensities, '-o');
xlabel('Time (minutes)');
ylabel('Mean intensity');
title('Timelapse — Mean Brightness');
