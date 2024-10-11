clc
clear all
close all

s = serialport("COM3","38400","DataBits",8,"StopBits",1);
configureTerminator(s,"CR");

% Read power
%writeline(s,"N1,00");
%power = readline(s);



%%Read rate
%start clock
%start loop
%send multi-tag read
multiTagTable = zeros(1,2);

for i = 1:20
    multiTagRead = writeline(s,"U"); % Multi Tag EPC command writeline is used to send ASCII 
    pause(0.1)  % delay to give time to serial command

    % read serial response
    readData = strings;
    tagId = strings;
    while s.NumBytesAvailable > 0
        line = s.readline();
        readData = [readData;line]; % Read returned tag ID's
        tagId = readData(3:end);
    end
        tagIdIndex = find(multiTagTable == tagID);
        if isempty(tagID_index)
            table_entry = [1,tagID];
            multiTagTable(end,1) = 1;
            multiTagTable(end,2) = tagId;
        end
    %find tag id in second column of table
    % if tag ID is in table
        % first column in tag ID row incremented by 1
    % if id is not in table, append new row to table, count = 1
        newline = [1,tagID]
        [multiTagTable]
%create a table to store tag id's and read counts


%create table: column 1 = numcounts, column = tag id
%stop loop
%stop clock
%for specific tag, divide numcounts by time
