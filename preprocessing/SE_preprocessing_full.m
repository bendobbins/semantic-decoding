%%
addpath("/projects/p33198/ben/semantic-decoding-ecog/preprocessing_full/");

%%
clear all
close all;
clc;

%% Set up subject-specific config
cfg.subjname = 'SE';
cfg.task = 'pictureNaming';
cfg.expnum = 1;

cfg.whitematter={
    'AC' + string([2:11]) ...
    'Q' + string([4:11]) ...
    'O' + string([10 15]) ...
    'M' + string([4]) ...
    'N' + string([12]) ...
    'F' + string([4]) ...
    'Y' + string([6:9 14:15]) ...
    'A' + string([7:9]) ...
    'B' + string([4:8]) ...
    'C' + string([3:6]) ...
    'S' + string([7]) ...
    'T' + string([4 6:9]) ...
    'L' + string([4:7]) ...
    'W' + string([2:13 15:16]) ...
    'DC03'% technically not white matter channel but should be removed as well
    };
cfg.whitematter = [cfg.whitematter{:}];

% Shanks or grids of electrodes to include in processing
% cfg.lead_of_interest = {'A','B','C','D','T','S','PC','R','L','K','J','X','Y','Z','O','AC'};
cfg.lead_of_interest = 'all';

% Frequency bands to process
cfg.freqs = {'HG'}; % ,'beta','alpha','theta'};

% Cropping bleed for epochs
cfg.crop_start_bleed = 1; % second before trial start
cfg.crop_end_bleed = 1; % second after max(trial end, voice offset)

% Missed trials for this specific subject/task/exp #
% These are also detected downstream so this step can be skipped
% cfg.missed_trials = [];
% if strcmp(cfg.task,'pictureNaming') && cfg.expnum==5
%     cfg.missed_trials = [1,152:153,190];
% elseif strcmp(cfg.task,'pictureNaming') && cfg.expnum==6
%     cfg.missed_trials = [131,175];
% end

% Whether to inspect all_data before filtering frequency bands and saving to file
cfg.flags.inspectBeforeSaving = false;
% Whether to normalize (to whole session mean/sd) signal before saving
cfg.flags.normalizeBeforeSaving = true;
% Save baseline statistics as well
cfg.flags.saveSessionBaseline = false;
cfg.flags.saveRestBaseline = false;

%% Load file(s)
[FileName, PathName, ~] = uigetfile('*.mat', 'MultiSelect', 'on');

if isequal(FileName,0)
    return
end
% Normalize to cell array
if ischar(FileName)
    FileName = {FileName};
end

for i = 1:numel(FileName)
    cfg.home_dir = fullfile(PathName,"..");
    cfg.data_path = fullfile(PathName, FileName{i});
    fprintf("Reading: %s\n", cfg.data_path)

    % Try to parse filename as: subj_task_expnum.mat (task may contain underscores)
    [~, base, ~] = fileparts(FileName{i});
    % toks = split(base, "_");
    expr = '^([^_]+)_(.+)_(\d+)$';
    toks = regexp(base, expr, 'tokens', 'once');

    % if numel(toks) == 3
    if ~isempty(toks)
        cfg.subjname = toks{1};
        cfg.task     = toks{2};
        % cfg.expnum   = str2double(toks{3});
        cfg.expnum   = toks{3};
    end
    fprintf("Subject: %s | Task: %s | Exp: %s\n", cfg.subjname, cfg.task, cfg.expnum)

    preprocess(cfg);
end