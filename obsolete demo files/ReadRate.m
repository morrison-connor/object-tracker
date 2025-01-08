clc
clear all
close all

s = serialport("COM3","38400","DataBits",8,"StopBits",1); % Configure serial port connection
configureTerminator(s,"CR/LF"); % Configure message send terminator characters

%multiTagTable = zeros(1,2); % Initialize array for storing tag ID's and read counts
TagList = strings;
counts = [];

numSend = 10; % Number of times multitag cammand is sent to reader

for i = 1:numSend
    writeline(s,"U"); % Multi Tag EPC command writeline is used to send ASCII 
    pause(0.1)  % delay to give time to serial command

    while s.NumBytesAvailable > 0 % While there are bytes in the message received
        line = s.readline(); 
        %readData = [readData;line]; % store tag ID in readData
        tagIDIndex = find(TagList == line,1);
        if isempty(tagIDIndex)
            TagList = [TagList;line];
            counts = [counts; 1];
        else
            counts(tagIDIndex) = counts(tagIDIndex) + 1;

        end
    end
end

disp("Tag List and Counts:")
disp(table(TagList,counts))

%Now we can perform a multitag read as many times as desired
%TagList consists of string values of all tag IDs read
