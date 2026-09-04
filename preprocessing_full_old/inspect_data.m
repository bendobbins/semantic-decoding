function data = inspect_data(data)
    openvar("data.all_data")
    % assignin('base', 'all_data', data.all_data) % copy to base workspace to make edits
    % openvar('all_data')
    % input("Edit data as required, then press Enter to continue.\n","s")
    fprintf("Edit data as required, then press Continue.\n")
    
    keyboard    % Make relevant modifications
    % data.all_data = evalin('base', 'all_data'); % write back
end

% function data = inspect_data(data)
%     fprintf("Edit data as required, then press Done.\n")
%     f = figure('Name','Edit data','NumberTitle','off');
%     t = uitable(f,'Data',data.all_data,...
%                 'ColumnEditable',true(1,size(data.all_data,2)),...
%                 'Position',[20 20 500 300]);
%     uicontrol(f,'Style','pushbutton','String','Done',...
%               'Callback', @(~,~) uiresume(f));
%     drawnow();
%     uiwait(f);
%     data.all_data = t.Data;
%     close(f)
% end