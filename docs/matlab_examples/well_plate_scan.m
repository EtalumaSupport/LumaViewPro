%% Well Plate Scan — Capture BF across a 96-well plate row
%
% Scans wells A1-A12, capturing BF at each position.
% Well positions are in plate coordinates (mm from top-left corner).

scope = lvp_connect();

scope.set_objective('10x Oly');
scope.set_exposure(2);
scope.set_gain(1.0);

%% Define well positions (96-well plate, row A)
% Standard 96-well: 9mm spacing, A1 center at (14.38, 11.24) mm
well_spacing_mm = 9.0;
a1_x_mm = 14.38;
a1_y_mm = 11.24;

n_wells = 12;
well_labels = arrayfun(@(i) sprintf('A%d', i), 1:n_wells, 'UniformOutput', false);

%% Convert plate mm to stage um
% Note: actual conversion depends on stage offset and labware config.
% These are approximate — the REST API will provide a coordinate
% transform endpoint for exact conversion.
um_per_mm = 1000;
stage_x = (a1_x_mm + (0:n_wells-1) * well_spacing_mm) * um_per_mm;
stage_y = repmat(a1_y_mm * um_per_mm, 1, n_wells);

%% Scan
images = cell(1, n_wells);
scope.led_on('BF', 100);

for i = 1:n_wells
    fprintf('Scanning well %s (%d/%d)...\n', well_labels{i}, i, n_wells);

    scope.move('X', stage_x(i), true);
    scope.move('Y', stage_y(i), true);

    images{i} = scope.capture();
end

scope.leds_off();

%% Display as montage
figure;
montage(images, 'Size', [1 n_wells]);
title('Row A — Brightfield');
