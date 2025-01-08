function readData = read_tags(s)
    writeline(s,"U"); % Multi Tag EPC command writeline is used to send ASCII 
    pause(0.1)  % delay to give time to serial command
    readData = strings;
    while s.NumBytesAvailable > 0 % While there are bytes in the message received
        line = s.readline();

        % code to remove 
        readData = [readData;line]; % store tag ID in readData as a string array
    end
end

