%%% Object tracker angle finder for demo purposes
clear

%% user inputs
tagId = '3000E2806894000040319034C93BD2F2';

nMeasurements = 100;
antennaCycles = 10;
windowSize = 6;  % how many data points to use in moving average
antennaPorts = [1 4];  % input antenna ports used ie. [1 2 4]
s = serialport('COM6', 38400);

%% setup
nAntennas = length(antennaPorts);
measurements = NaN(nAntennas, windowSize);
weightedAvg = NaN(1, nAntennas);
s.configureTerminator("CR/LF");

%% main loop
measurementNum = 0;
totalRead = zeros(1, 2);
while(true)
    %input("Press any key to start")
    % 1. take a read ratio measurement
    disp("1. take a read ratio measurement")
    measurementNum = measurementNum + 1;
    for ii = 1:floor(nMeasurements/(nAntennas*antennaCycles))
        for jj = 1:nAntennas

            % a. switch antenna
            disp("a. switch antenna")
            antennaNum = antennaPorts(jj);
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
            for kk = 1:length(cmd)
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
            disp("b. take measurements")
            for kk = 1:antennaCycles
                % run RFID reader
                readData = read_tags(s);
                % check if target tag was detected
                if any(contains(tagId, readData))  % looks for tagId in all returned data
                    totalRead(jj) = totalRead(jj) + 1;
                end
            end
            fprintf("Total measurements = %d\n", sum(totalRead))
        end
    end

    % 2. calculate the read ratio
    disp("2. calculate the read ratio")
    for jj = 1:nAntennas
        readRatio = totalRead(jj) / nMeasurements;
        fprintf("Antenna %d readRatio = %0.2f\n", antennaPorts(jj), readRatio)

        % weightedAvg across windowSize
        if measurementNum < windowSize
            measurements(jj, measurementNum) = readRatio;
            weightedAvg(jj) = mean(measurements(jj, 1:measurementNum));
        else
            measurements(1:end-1) = measurements(2:end);
            measurements(jj, end) = readRatio;
            weightedAvg(jj) = mean(measurements(jj));
        end
    end

    % 3. use weightedAvg to determine position (for a 2 antenna system)
    fprintf("Weighted Avg 1 = %0.2f\n", weightedAvg(1))
    fprintf("Weighted Avg 2 = %0.2f\n", weightedAvg(2))
    disp("3. use weightedAvg to determine position")
    delta = weightedAvg(1) - weightedAvg(2);
    if delta > 0
        disp("Turn Right")
    elseif delta < 0
        disp("Turn Left")
    else
        disp("Correct direction")
    end
    pause(2);
end
