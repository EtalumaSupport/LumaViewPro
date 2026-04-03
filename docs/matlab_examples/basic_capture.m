%% Basic Capture — Move, illuminate, capture, display
%
% Moves to a position, turns on BF LED, captures an image, and displays it.
% This is the simplest possible use case.

% Connect to LumaViewPro
scope = lvp_connect();

% Check system is ready
status = scope.status();
fprintf('Connected: LED=%d Motor=%d Camera=%d\n', ...
    status.led_connected, status.motor_connected, status.camera_connected);

% Set objective and camera
scope.set_objective('10x Oly');
scope.set_exposure(50);    % 50 ms
scope.set_gain(5.0);       % 5 dB

% Move to position (um)
scope.move('X', 60000, true);   % blocking
scope.move('Y', 40000, true);
scope.move('Z', 5000, true);

% Illuminate and capture
scope.led_on('BF', 100);        % Brightfield at 100 mA
img = scope.capture();
scope.leds_off();

% Display
figure;
imshow(img);
title('Brightfield Capture');

% Save
result = scope.save_image('C:\data\bf_capture.tiff');
fprintf('Saved to: %s\n', result.file_path);
