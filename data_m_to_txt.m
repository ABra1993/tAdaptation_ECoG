%% subject list
subjects = ["sub-p11", "sub-p12", "sub-p13", "sub-p14"];
subject_num = numel(subjects);
% disp(subject_num)

dirData = ' ' %%%%%%%%%% DIRECTORY of source files (.m)
dirSave = ' ' %%%%%%%%%% DIRECTORY of output (.txt)

for j = 1:length(subjects)
        
    % select subject
    subject = subjects{j};
    fprintf(strcat('Computing for', 32,  subject, '...\n'));
    
    dirSave_subject = strcat(dirSave, subject);

    FileName = strcat(dirData, subject, '_data_nobaselinecorrection.mat');
    FileData = load(FileName);

    % Extract DataTables
    t = FileData.t;
    channels = FileData.channels;
    electrodes = FileData.electrodes;   
    events = FileData.events;
    epochs_b = FileData.epochs_bb;
    epochs_v = FileData.epochs_vt;

    % Write tables to .txt files
    writematrix(t, strcat(dirSave_subject, '/t.txt'));
    writetable(channels, strcat(dirSave_subject, '/channels.txt'));
    writetable(electrodes, strcat(dirSave_subject, '/electrodes.txt'));
    writetable(events, strcat(dirSave_subject, '/events.txt'));

    % write epochs_b to .txt file
    channels_num = size(channels, 1);
    for i = 1:channels_num
        table = epochs_b(:, :, i);
        writematrix(table, strcat(dirSave_subject, '/epochs_b/epochs_b_channel', num2str(i),'.txt'));
    end

%     % write epochs_v to .txt file
%     for i = 1:channels_num
%         table = epochs_v(:, :, i);
%         writematrix(table, strcat(dirSave_subject, '/epochs_v/epochs_v_channel', num2str(i),'.txt'));
%     end
    
end
