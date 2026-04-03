classdef lvp_connect < handle
    % LVP_CONNECT  MATLAB interface to LumaViewPro REST API
    %
    %   scope = lvp_connect();                    % localhost:8000
    %   scope = lvp_connect('192.168.1.100');     % remote host
    %   scope = lvp_connect('localhost', 9000);   % custom port
    %
    %   scope.led_on('BF', 100);
    %   scope.move('Z', 5000, true);
    %   img = scope.capture();
    %   scope.leds_off();
    %
    % Requires MATLAB R2016b+ for webread/webwrite.

    properties (Access = private)
        base_url
        options
    end

    methods
        function obj = lvp_connect(host, port)
            % Constructor — connect to LumaViewPro REST API
            if nargin < 1; host = 'localhost'; end
            if nargin < 2; port = 8000; end
            obj.base_url = sprintf('http://%s:%d/api/v1', host, port);
            obj.options = weboptions('ContentType', 'json', 'Timeout', 30);
        end

        %% System
        function s = status(obj)
            % Get system status (connected hardware, protocol state, etc.)
            s = webread([obj.base_url '/status'], obj.options);
        end

        function info = system_info(obj)
            % Get microscope model, serial number, firmware versions
            info = webread([obj.base_url '/system/info'], obj.options);
        end

        %% LED Control
        function led_on(obj, color, mA)
            % Turn on an LED channel
            %   scope.led_on('BF', 100)     % Brightfield at 100 mA
            %   scope.led_on('Blue', 200)   % Blue fluorescence at 200 mA
            data = struct('color', color, 'mA', mA);
            webwrite([obj.base_url '/led/on'], data, obj.options);
        end

        function led_off(obj, color)
            % Turn off a specific LED channel
            %   scope.led_off('BF')
            data = struct('color', color);
            webwrite([obj.base_url '/led/off'], data, obj.options);
        end

        function leds_off(obj)
            % Turn off all LEDs
            webwrite([obj.base_url '/led/off_all'], '', obj.options);
        end

        function s = led_states(obj)
            % Get current LED states for all channels
            s = webread([obj.base_url '/led/states'], obj.options);
        end

        %% Motion Control
        function move(obj, axis, position_um, wait)
            % Move an axis to an absolute position
            %   scope.move('Z', 5000)           % non-blocking
            %   scope.move('Z', 5000, true)     % wait until complete
            %   scope.move('X', 60000, true)
            if nargin < 4; wait = false; end
            data = struct('axis', axis, 'position_um', position_um, ...
                          'wait', wait);
            webwrite([obj.base_url '/motion/move'], data, obj.options);
        end

        function move_relative(obj, axis, um, wait)
            % Move an axis by a relative amount
            %   scope.move_relative('Z', 100)    % up 100 um
            %   scope.move_relative('X', -500)   % left 500 um
            if nargin < 4; wait = false; end
            data = struct('axis', axis, 'um', um, 'wait', wait);
            webwrite([obj.base_url '/motion/move_relative'], data, obj.options);
        end

        function home(obj)
            % Home all axes (XY, Z, turret)
            webwrite([obj.base_url '/motion/home'], '', obj.options);
        end

        function pos = get_position(obj, axis)
            % Get current position of an axis (um)
            %   z = scope.get_position('Z')
            pos = webread([obj.base_url '/motion/position/' axis], obj.options);
        end

        function pos = get_positions(obj)
            % Get positions of all axes
            pos = webread([obj.base_url '/motion/positions'], obj.options);
        end

        function tf = is_moving(obj)
            % Check if any axis is currently moving
            s = webread([obj.base_url '/motion/status'], obj.options);
            tf = s.is_moving;
        end

        function wait_until_stopped(obj, timeout_s)
            % Block until all axes stop moving
            if nargin < 2; timeout_s = 30; end
            t0 = tic;
            while toc(t0) < timeout_s
                if ~obj.is_moving(); return; end
                pause(0.1);
            end
            warning('lvp_connect:timeout', 'Timed out waiting for motion');
        end

        %% Camera Control
        function set_exposure(obj, ms)
            % Set camera exposure time in milliseconds
            %   scope.set_exposure(50)
            data = struct('exposure_ms', ms);
            webwrite([obj.base_url '/camera/exposure'], data, obj.options);
        end

        function set_gain(obj, dB)
            % Set camera gain in dB
            %   scope.set_gain(10.0)
            data = struct('gain_dB', dB);
            webwrite([obj.base_url '/camera/gain'], data, obj.options);
        end

        function img = capture(obj, opts)
            % Capture a single frame (frame-validity aware)
            %   img = scope.capture()
            %   img = scope.capture(struct('format','tiff','save',true))
            %
            % Returns image as MATLAB matrix (uint8 or uint16).
            % If save=true, also saves to disk and returns file path in
            % result.file_path.
            if nargin < 2; opts = struct(); end
            result = webwrite([obj.base_url '/capture'], opts, obj.options);

            if isfield(result, 'file_path') && ~isempty(result.file_path)
                img = imread(result.file_path);
            elseif isfield(result, 'image_base64')
                % Decode base64 image data
                bytes = matlab.net.base64decode(result.image_base64);
                img = imdecode(bytes);
            else
                img = [];
                warning('lvp_connect:no_image', 'No image data in response');
            end
        end

        function frame = live_frame(obj)
            % Grab a single live frame (no frame validity, fast)
            %   frame = scope.live_frame()
            opts_bin = weboptions('ContentType', 'binary', 'Timeout', 5);
            bytes = webread([obj.base_url '/camera/live_frame'], opts_bin);
            frame = reshape(typecast(bytes, 'uint8'), [], []);
        end

        function info = camera_info(obj)
            % Get camera model, resolution, gain/exposure ranges
            info = webread([obj.base_url '/camera/info'], obj.options);
        end

        %% Objective
        function set_objective(obj, objective_id)
            % Set the active objective
            %   scope.set_objective('10x Oly')
            data = struct('objective_id', objective_id);
            webwrite([obj.base_url '/objective'], data, obj.options);
        end

        function info = get_objective(obj)
            % Get current objective info (magnification, NA, etc.)
            info = webread([obj.base_url '/objective'], obj.options);
        end

        %% Protocol
        function run_protocol(obj, file_path)
            % Run a protocol from a .tsv file
            %   scope.run_protocol('C:\protocols\my_protocol.tsv')
            data = struct('file_path', file_path);
            webwrite([obj.base_url '/protocol/run'], data, obj.options);
        end

        function abort_protocol(obj)
            % Abort a running protocol
            webwrite([obj.base_url '/protocol/abort'], '', obj.options);
        end

        function s = protocol_status(obj)
            % Get protocol execution status
            s = webread([obj.base_url '/protocol/status'], obj.options);
        end

        %% Image Saving
        function result = save_image(obj, file_path, format)
            % Save the last captured image to a file
            %   scope.save_image('C:\data\image.tiff')
            %   scope.save_image('C:\data\image.ome.tiff', 'OME-TIFF')
            if nargin < 3; format = 'TIFF'; end
            data = struct('file_path', file_path, 'format', format);
            result = webwrite([obj.base_url '/image/save'], data, obj.options);
        end
    end
end
