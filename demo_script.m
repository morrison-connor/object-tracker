%%% Object tracker angle finder for demo purposes
clear

%% user inputs

% tag lookup
card = '3000E2806894000040319034C93BD2F2';
rect = '3000E280691500007003C56364E08DD3';
square = '3400E280F3362000F00005E60D18E39C';

tagId = square;

nMeasurements = 1000;
antennaCycles = 100;
windowSize = 6;  % how many data points to use in moving average
antennaPorts = [1 2];  % input antenna ports used ie. [1 2 4]
s = serialport('COM6', 38400);
tolerance = 0.04;

%% setup
nAntennas = length(antennaPorts);
measurements = NaN(nAntennas, windowSize);
readRatio = NaN(1, nAntennas);

s.configureTerminator("CR/LF");

%% Initialize:
% Open antenna ports N0 and N1:
s.writeline("N7,20");
s.writeline("N7,10");
s.writeline("N7,11");
% Commands for switching antennas:
s.writeline("N9,20");
N0cmd = "N9,10";
N1cmd = "N9,11";
s.writeline(N0cmd);

%% main loop
measurementNum = 0;
while(true)
    totalSend = 0;
    totalRead = zeros(1, 2);
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
                    cmd = N0cmd;
                case 2
                    cmd = N1cmd;
                case 3
                    cmd = ["N9,N22", "N9,11"];
                case 4
                    cmd = ["N9,N22", "N9,10"];
            end
            check = "";
            while ~strcmp(check, "N"+string(antennaNum - 1))
                s.writeline(cmd)
                pause(0.1);
                check = s.readline();
                check = char(check);
                check = check(2:end);
            end
            fprintf("antenna %d selected\n", antennaNum)

            % b. take measurements
            disp("b. take measurements")
            for kk = 1:antennaCycles
                % run RFID reader
                readData = read_tags(s);
                totalSend = totalSend + 1;
                % check if target tag was detected
                if any(contains(readData, tagId))  % looks for tagId in all returned data
                    totalRead(jj) = totalRead(jj) + 1;
                end
            end
            fprintf(['Total read: ' repmat(' %1.0f ',1,numel(totalRead)) '\n'],totalRead);
            fprintf("Total signals sent = %d\n", totalSend)
        end
    end

    % 2. calculate the read ratio
    disp("2. calculate the read ratio")
    for jj = 1:nAntennas
        readRatio(jj) = totalRead(jj) / (nMeasurements/nAntennas);
        fprintf("Antenna %d readRatio = %0.2f\n", antennaPorts(jj), readRatio(jj))
    end

    fprintf("\n");
    delta = readRatio(1) - readRatio(2);
    if delta > tolerance
        disp("Turn Right")
    elseif delta < -tolerance
        disp("Turn Left")
    else
        disp("Correct direction")
    end
    pause(0.5);
    fprintf("\n");
end
