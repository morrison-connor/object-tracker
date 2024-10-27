%%% Object tracker angle finder for demo purposes

%% user inputs
targetId = '3000E2806894000040319034C93BD2F2';

nMeasurements = 200;
antennaCycles = 10;
windowSize = 6;  % how many data points to use in moving average
antennaPorts = [1 2];  % input antenna ports used ie. [1 2 4]
s = serialport('COM6', 38400);

%% setup
nAntennas = len(antennaPorts);
measurements = NaN(nAntennas, windowSize);
weightedAvg = NaN(1, nAntennas);
s.configureTerminator("CR/LF");



    writeline(s,"U"); % Multi Tag EPC command writeline is used to send ASCII 
    pause(0.1)  % delay to give time to serial command
    readData = strings;
    while s.NumBytesAvailable > 0 % While there are bytes in the message received
        line = s.readline();
        readData = [readData;line]; % store tag ID in readData
    end



%% main loop
measurementNum = 0;
totalRead = zeros(1, 2);
while(true)
    % 1. take a read ratio measurement
    measurementNum = measurementNum + 1;
    for jj = 1:nMeasurements
        for kk = 1:nAntennas

            % a. switch antenna
            antennaNum = antennaPorts(kk);
            switch antennaNum
                case 1
                    cmd = ["N9,N20", "N9,10"];
                case 2
                    cmd = ["N9,N20", "N9,11"];
                case 3
                    cmd = ["N9,N22", "N9,11"];
                case 4
                    cmd = ["N9,N22", "N9,10"];
            end
            for ii = 1:len(cmd)
               s.writeline(cmd(1));  % Send 1st antenna switch command
               pause(0.1)
               s.readline();
               pause(0.1);
               s.writeline(cmd(2));  % Send 2nd antenna switch command
               pause(0.1);
               s.readline();
               % do we need to switch antenna open??
               % <LF>N7,22<CR>, <LF>N7,11<CR>
            end

            % b. take measurements
            for ll = 1:antennaCycles
                % run RFID reader
                readData = read_tags(s);
                % check if target tag was detected
                if any(contains(tagId, readData))  % looks for tagId in all returned data
                    totalRead(kk) = totalRead(kk) + 1;
                end
            end
        end
    end

    % 2. calculate the read ratio
    for kk = 1:nAntennas
        readRatio = totalRead(kk) / nMeasurements;
        % weightedAvg across windowSize
        if measurementNum < windowSize
            measurements(kk, measurementNum) = readRatio;
            weightedAvg(end) = mean(measurements(kk, 1:measurementNum));
        else
            measurements(1:end-1) = measurements(2:end);
            measurements(kk, end) = readRatio;
            weightedAvg(end) = mean(measurements(kk));
        end
    end

    % 3. use weightedAvg to determine position
    delta = weightedAvg(1) - weightedAvg(2);
    if delta > 0
        disp("Turn Right")
    elseif delta < 0
        disp("Turn Left")
    else
        disp("Correct direction")
    end
end
