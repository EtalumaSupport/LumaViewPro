%% Multi-Channel Capture — BF + Fluorescence composite
%
% Captures BF and multiple fluorescence channels at the same position,
% then builds a composite overlay.

scope = lvp_connect();

scope.set_objective('10x Oly');
scope.move('X', 60000, true);
scope.move('Y', 40000, true);
scope.move('Z', 5000, true);

%% Capture BF (transmitted light)
scope.set_exposure(2);
scope.set_gain(1.0);
scope.led_on('BF', 100);
img_bf = scope.capture();
scope.leds_off();

%% Capture Blue fluorescence
scope.set_exposure(100);
scope.set_gain(15.0);
scope.led_on('Blue', 200);
img_blue = scope.capture();
scope.leds_off();

%% Capture Green fluorescence
scope.set_exposure(80);
scope.set_gain(12.0);
scope.led_on('Green', 150);
img_green = scope.capture();
scope.leds_off();

%% Capture Red fluorescence
scope.set_exposure(90);
scope.set_gain(10.0);
scope.led_on('Red', 180);
img_red = scope.capture();
scope.leds_off();

%% Build composite in MATLAB
% BF as grayscale base, fluorescence overlaid in color
composite = repmat(img_bf, [1 1 3]);  % grayscale to RGB

% Overlay fluorescence channels with brightness thresholding
threshold = 20;  % pixels below this stay as BF

blue_mask = img_blue > threshold;
composite(:,:,3) = max(composite(:,:,3), uint8(blue_mask) .* img_blue);

green_mask = img_green > threshold;
composite(:,:,2) = max(composite(:,:,2), uint8(green_mask) .* img_green);

red_mask = img_red > threshold;
composite(:,:,1) = max(composite(:,:,1), uint8(red_mask) .* img_red);

%% Display
figure;
subplot(2,3,1); imshow(img_bf);    title('BF');
subplot(2,3,2); imshow(img_blue);  title('Blue');
subplot(2,3,3); imshow(img_green); title('Green');
subplot(2,3,4); imshow(img_red);   title('Red');
subplot(2,3,[5 6]); imshow(composite); title('Composite');
