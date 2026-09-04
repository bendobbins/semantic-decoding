function data = prepare_data(cfg)
    clear data

    %% Load data
    % Main neural/behavioral data (signal/states/parameters)
    load(cfg.data_path);
    clinicalFs = parameters.SamplingRate.NumericValue;
    ECoGDataMatrix = double(signal);
    
    % Bad trial/channel information
    bads_path = insertBefore(cfg.data_path, '.mat', '_bad');
    if isfile(bads_path)
        load(bads_path, 'info')
        data.bad_channels = info.bad_channels;
        data.bad_trials = info.bad_trials;
        data.car_info = info.CAR_info;
    else
        fprintf("Bad channels/trials file not found. Continuing without this data\n")
        % info = struct;
        data.bad_channels = []; % {};
        data.bad_trials = []; % {};
        data.car_info =  {};
    end
    
    % Audio annotations
    % Search a few known layouts/naming conventions: the labels may live in a
    % <subj>_Audio subfolder or flat alongside the .mat/.wav files in home_dir.
    lbl_dir = fullfile(cfg.home_dir, "data/labels");
    lbl_base = sprintf("%s_%s_%s", cfg.subjname, cfg.task, cfg.expnum);
    lbl_dirs = [string(lbl_dir), string(cfg.home_dir)];
    lbl_suffixes = ["_labels.txt", "_in_labels.txt", ".txt", "_in.txt", "_story_words.txt"];

    audio_labels_path = fullfile(lbl_dir, lbl_base + "_labels.txt"); % default, for the message below
    found_labels = false;
    for lbl_d = 1:numel(lbl_dirs)
        for lbl_s = 1:numel(lbl_suffixes)
            lbl_candidate = fullfile(lbl_dirs(lbl_d), lbl_base + lbl_suffixes(lbl_s));
            if isfile(lbl_candidate)
                audio_labels_path = lbl_candidate;
                found_labels = true;
                break
            end
        end
        if found_labels
            break
        end
    end

    if found_labels
        fprintf("Loading audio labels from: %s\n", audio_labels_path);
        voice_tbl = readtable(audio_labels_path, ...
            'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);
        fprintf("Loaded %d annotations\n", height(voice_tbl));
    else
        fprintf("Audio labels file not found. Continuing without voice data\n");
    end

    %% Parse channel names
    try
        names = regexprep(parameters.ChannelNames.Value, '(?<=[Ee][Mm][Gg])\s+(?=\d+-REF)', '');
        names = regexprep(names, '(?<=[Ee][Kk][Gg])\s+(?=\d+-REF)', '');
        names = regexp(names, '\S+', 'match');
        names = cellfun(@(x) regexprep(x, '\*', ''), names, 'UniformOutput', false);
        channelIDs = string(cellfun(@(x) regexprep(x, '[-_]REF-?\d*$', '', 'ignorecase'),names,'UniformOutput', false));  % remove '-REF' at end
    catch
        channelIDs = arrayfun(@(x) sprintf('Chn %d', x), 1:size(ECoGDataMatrix,2), 'UniformOutput', false);
    end
    channelIDs = reshape(channelIDs, 1, []);
    
    %% Process event timings
    % Here we also want to account for delay caused by using a FIR filter on neural signal later
    firorder = floor((4/10) * clinicalFs);
    fir_delay = floor((firorder-1)/2);
    fir_delay_s = fir_delay / clinicalFs;

    %seq = int64(states.StimulusCode); % [T x 1] indices over time (0 = none)
    %vals = parameters.Stimuli.Value; % 3 x N cell, row 2 are file paths
    %[onsets, classes, map] = find_stim_onsets_from_sequence(seq, vals);
    
    % Trial end: use end of go cue if it exists, else end of stim
    %if all(isnan(onsets.go))
    %    trial_end = (onsets.stim_end + fir_delay);
    %else
    %    trial_end = (onsets.go_end + fir_delay);
    %end

    % trials are now just when the word is spoken, so end is when the voice
    % stops
    %trial_end = (voice_tbl.Var2 + fir_delay_s);

    % If recording ended mid-trial there may be errors finding onsets
    % (particularly trial ends). Remove last trial if this occurs
    %if isnan(trial_end(end))
        %fields = fieldnames(onsets);
        % Remove last element of each array in onsets
        %for i = 1:length(fields)
            %field = fields{i};
            %onsets.(field)(end) = [];
        %end
    %end

    % Process stimulus timings and values
    %stim_order = onsets.codes_stim;
    %trial_start = (voice_tbl.Var1 + fir_delay_s) / clinicalFs;
    %trial_end = trial_end / clinicalFs;

    % Process go cue timings, if exist
    %if ~any(isnan(onsets.go_end))
    %    go_cue = (onsets.go + fir_delay) / clinicalFs;
    %    go_cue_rel = go_cue - trial_start;
    %end
    
    % Process fixation cross timings, if exist
    %if ~any(isnan(onsets.fixation))
    %    fixation = (onsets.fixation + fir_delay) / clinicalFs;
    %    fixation_rel = fixation - trial_start;
    %end
    
    % Process voice times and labels   
    if exist("voice_tbl","var")
        answered_words = voice_tbl.Var3;
        voice_onset_all = voice_tbl.Var1 + fir_delay_s;
        voice_offset_all = voice_tbl.Var2 + fir_delay_s;
        % trial_end = voice_offset_all' + 1;
    end
    
    %% Process auditory prompt durations (auditory naming task only)
    if strcmp(cfg.task, "auditoryNaming")
        auditory_prompts_path = fullfile( ...
            cfg.home_dir, ...
            sprintf("%s_AuditoryNamingStimuli/wavs", cfg.subjname) ...
        );
        % 'UseSavedOutputs' option controls whether or not to actually process .wavs with Whisper:
        % If this is the first time, set to false
        % If json files for .wav transcriptions are already stored, set to true
        durations = get_wav_durations( ...
            auditory_prompts_path, map.labels(classes.stimulus_codes), ...
            'CondaEnv', 'whisper-audio', ...
            'OutputDir', replace(auditory_prompts_path, 'wavs', 'durations') ...
        );
    end

    %% Filter line noise from neural signal
    f0=60; % Hz
    nyquist = clinicalFs / 2;
    harmonics = f0:f0:nyquist;
    Q = 30; % Quality factor (adjust as needed for sharpness of the filter)
    for h = harmonics
        bandwidth = h / Q;
        if h+bandwidth/2>clinicalFs/2
            continue
        end
        notchFilter = designfilt('bandstopiir', ...
            'FilterOrder', 4, ...
            'HalfPowerFrequency1', h - bandwidth / 2, ...
            'HalfPowerFrequency2', h + bandwidth / 2, ...
            'SampleRate', clinicalFs, ...
            'DesignMethod', 'butter');
    
        ECoGDataMatrix = filtfilt(notchFilter, ECoGDataMatrix); % first dimension is time, second dimension is n channels
    end
    
    %% Common average referencing (CAR)
    exclude_leads = {'DC'};
    if ~isfield(cfg, "lead_of_interest") || ~any(strcmp(cfg.lead_of_interest, "all"))
        lead_of_interest = cfg.lead_of_interest;
    else
        lead_of_interest = unique(regexprep(channelIDs,'\d+',''), 'stable');
    end
    lead_of_interest = lead_of_interest(~ismember(lead_of_interest, exclude_leads));
    
    badchannel_names = channelIDs(data.bad_channels);
    concatenated_data = [];
    channels_all = [];
    channelIdxs_all = [];

    for leadcount = 1:length(lead_of_interest)
        channels = channelIDs;
        pattern = sprintf("^%s\\d+", lead_of_interest{leadcount});
        leadindex = find(cellfun(@isempty, regexp(channels, pattern))==0);
        % leadindex = data.car_info{leadcount};
        channels = channels(leadindex); % Channels as string labels
        channelIdxs = leadindex;

        % Channel names: replace prefix and reindex
        % nums = double(extractAfter(channels, 'chan'));
        % if ~any(isnan(nums))
        %     channels = lead_of_interest{leadcount} + string(nums - nums(1) + 1);
        % end
    
        removewhitematter = ismember(channels,cfg.whitematter);
        % removebadchannels = ismember(channels,badchannel_names);
        removebadchannels = ismember(channelIdxs, data.bad_channels);
    
        fprintf("\n=== Processing lead %s ===", lead_of_interest{leadcount});
        fprintf("\nWhite matter electrodes excluded from CAR: ");
        fprintf("%s, ", channels{removewhitematter});
        fprintf("\nBad electrodes excluded from CAR: ");
        fprintf("%s, ", channels{removebadchannels});
        fprintf("\n");
    
        % Save a copy of channels where bad channels are still kept (no change to CAR)
        % in case bad channels need to be collectively removed when concatenating trials across expts later
        leadindex_withbadchannels = leadindex;
        leadindex_withbadchannels(removewhitematter) = [];
        leadindex(removewhitematter | removebadchannels) = [];
        channels(removewhitematter) = [];
        channelIdxs(removewhitematter) = [];
    
        % Some edge cases for CAR
        % No good channels in lead => CAR will result in NaN => don't CAR
        % Only 1 good channel in lead => CAR will result in 0 => don't CAR
        if length(leadindex) <= 1
            processedECoG = ECoGDataMatrix(:, leadindex_withbadchannels);
        else  % Else, perform CAR, excluding bad + white matter channels from average, but retaining bad channels in final output
            commonAverage = mean(ECoGDataMatrix(:, leadindex), 2);
            % processedECoG = ECoGDataMatrix(:, leadindex);
            processedECoG = ECoGDataMatrix(:, leadindex_withbadchannels) - commonAverage;
        end
        
        concatenated_data = [concatenated_data processedECoG]; % stack across columns (channels)
        channels_all = [channels_all channels];
        channelIdxs_all = [channelIdxs_all channelIdxs];
        clear processedECoG commonAverage;
    end
    
    % Ultimately, want to remove white matter channels but retain bad channels - so diff bad channels across expts can be aggregated
    % => Reindex bad channel indices to (all channels - white matter channels). If a bad channel is also white matter, it is removed.
    % badchannels_reindexed = find(ismember(channels_all, badchannel_names));
    badchannels_reindexed = find(ismember(channelIdxs_all, data.bad_channels));
    data.ECoGDataMatrix_final = concatenated_data;
    
    %% Account for missed trials, if provided 
    %if isfield(cfg,"missed_trials")
    %    trial_start(cfg.missed_trials) = [];
    %    go_cue(cfg.missed_trials) = [];
    %    target(cfg.missed_trials) = [];
    %    bad_trials = data.bad_trials(~ismember(data.bad_trials, cfg.missed_trials));
    %    bad_trials = bad_trials - sum(cfg.missed_trials' < bad_trials);
    %else
    %    bad_trials = data.bad_trials;
    %end
    
    %% Prepare all_data structure with all fields EXCEPT epoched ECoG data
    numtrials = size(voice_tbl.Var3, 1);
    %goodMask = true(numtrials, 1);
    %try
    %    goodMask(cell2mat(bad_trials)) = false;
    %catch
    %    goodMask(bad_trials) = false;
    %end
    
    all_data = {};
    end_time = [];
    voice_onset = [];
    voice_offset = [];
    voice_onset_thresh = 0.8; % min response delay before voice can be considered within-trial
    
    for i = 1:numtrials
        all_data(i,2) = {0 + cfg.crop_start_bleed}; % voice onset
        all_data(i,3) = {voice_offset_all(i) + cfg.crop_end_bleed - voice_onset_all(i) + cfg.crop_start_bleed}; % voice offset
        all_data(i,4) = {answered_words(i)}; % answered word label
        voice_onset(i) = voice_onset_all(i);
        voice_offset(i) = voice_offset_all(i);
    
        all_data(i,1) = {[]}; % placeholder for epoched data
    
        all_data(i,5) = {badchannels_reindexed}; % bad channels
        %all_data(i,6) = {goodMask(i)}; % bad trials
        all_data(i,6) = {clinicalFs};
        all_data(i,7) = {convertStringsToChars(channels_all)}; % channel names
        %if exist("fixation_rel", "var")
        %    all_data(i,13) = {fixation_rel(i) + cfg.crop_start_bleed}; % fixation cross onset
        %end

        % Fields specific to other tasks
        %if strcmp(cfg.task, "auditoryNaming")
        %    target = map.labels(stim_order(i));
        %    words = durations(target).words;
            % words = table2struct(stimulus.words);
        %    all_data(i,14) = {map.types(i)};
        %    all_data(i,15) = {convertStringsToChars(words.word)}; % words in stimulus (auditory description)
        %    all_data(i,16) = {words.onset_s + cfg.crop_start_bleed}; % onset of the words in the stimulus
        %    all_data(i,17) = {words.offset_s + cfg.crop_start_bleed}; % offset of the words in the stimulus
        %elseif strcmp(cfg.task, "pictureNaming_wordsAloud")
        %    all_data(i,14) = {map.types(classes.stimulus_codes(i))}; % stimulus type (word or picture)
        %end
    end
    % At this point, we can inspect the structure to ensure all responses, bad trials etc. are appropriately labeled/aligned
    fprintf("\nTotal trials for this expt: %d\n", numtrials)

    data.all_data = all_data;
    data.clinicalFs = clinicalFs;
    data.voice_onset = voice_onset;
    data.voice_offset = voice_offset;
end