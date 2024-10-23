%%% Rough work for object tracker angle finder

%% user input
targetEpc = '';
antennaDataPath = '';

nMeasurements = 200;
windowSize = 6;  % how many data points to use in moving average
measurements = NaN(1, windowSize);

nAntennas = 1;
weightedAvg = NaN(1, nAntennas);





%% main loop
totalRead = zeros(1, 2);
for ii = 1:windowSize

    % take a readRatio measurement
    for jj = 1:nMeasurements
        for kk = 1:nAntennas
            % function here
            
    
            % check for received signal through serial
            if strcmp(tagId, targetEpc)
                totalRead(kk) = totalRead(kk) + 1;
            end
        end
    end
    readRatio = totalRead / nMeasurements;
    measurements(ii) = readRatio;
    
    % weightedAvg across windowSize
    if ii >= windowSize
        currentWindow = measurements(ii-windowSize+1:ii); % Get the latest 'windowSize' measurements
        weightedAvg = mean(currentWindow); % Compute the moving average
    end

    % use weightedAvg to determine position
    delta = weightedAvg(1) - weightedAvg(2);
    if delta > 0
        disp("Turn Right")
    elseif delta < 0
        disp("Turn Left")
    else
        disp("Correct direction")
    end
end
