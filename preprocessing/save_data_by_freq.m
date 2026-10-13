function save_path = save_data_by_freq(data, cfg)
    % Saves a separate all_data struct to file for each freq band specified.
    processedECoG = data.ECoGDataMatrix_final(:,find(sum(data.ECoGDataMatrix_final~=0)));
    clinicalFs = data.clinicalFs;
    all_data = data.all_data;

    % Define baseline-specific constants
    % These are a first pass for HG band for now, diff freqs may require diff durations
    cfg.baseline.buffer_post = 0.7;   % seconds after voice offset
    cfg.baseline.buffer_pre  = 0;   % seconds before next trial onset
    cfg.baseline.min_dur     = 0.3;   % minimum baseline duration (seconds)

    for freqcount = 1:length(cfg.freqs)
        clear hg_ECoG; all_data(:,1) = {[]};
        freq2use = cfg.freqs{freqcount};
        fprintf("Processing freq band: %s\n", freq2use)
        % Extract band power
        smoothed_hg_ECoG = extract_bandpower(processedECoG, clinicalFs, freq2use);
        data.bandpower = smoothed_hg_ECoG; % Note this gets overwritten
        
        % Extract baseline segments (experimental)
        if cfg.flags.saveRestBaseline
            baseline_rest = get_rest_baseline(data, cfg);
        end
        if cfg.flags.saveSessionBaseline
            baseline_session = struct('mu', {}, 'sigma', {});
            [~, mu_session, sigma_session] = zscore(smoothed_hg_ECoG,[],1);
            for i = 1:length(mu_session)
                baseline_session(i).mu = mu_session(i);
                baseline_session(i).sigma = sigma_session(i);
            end
        end

        % Here we zscore the entire file since there is no baseline period
        % Do this downstream if not now
        try
            if cfg.flags.normalizeBeforeSaving
                smoothed_hg_ECoG = zscore(smoothed_hg_ECoG, [], 1);
            end
        catch
        end
    
        % Segment into trials/epochs, and insert into all_data    
        fprintf("Epoching trials...\n")
        trials_outofbounds = [];
        % Crop the trial
        trial = smoothed_hg_ECoG(:, :);
        all_data(1,1) = {trial}; % freq band data
        all_data(trials_outofbounds, :) = [];
        
        % Save
        fprintf("Saving...\n")
        save_dir = fullfile(cfg.home_dir, "..", sprintf("%s_Processed/FIR", cfg.subjname));
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        save_path = fullfile( ...
            save_dir, ...
            sprintf("%s_%s_%s_%s_all_data.mat",cfg.subjname,cfg.task,cfg.expnum,freq2use) ...
        );
        save(save_path, "all_data", "-v7.3")
        fprintf("\nData saved to: %s\n", save_path);

        if cfg.flags.saveRestBaseline & numel(baseline_rest) > 0 % Save if any rest segments were found
            baseline_rest_path = replace(save_path, "all_data", "baseline_rest");
            save(baseline_rest_path, "baseline_rest", "-v7.3")
            fprintf("\nBaseline data saved to: %s\n", baseline_rest_path);
        end
        if cfg.flags.saveSessionBaseline
            baseline_session_path = replace(save_path, 'all_data', 'baseline_session');
            save(baseline_session_path, 'baseline_session', '-v7.3')
            fprintf("\nBaseline data saved to: %s\n", baseline_session_path);
        end
    end
end


function smoothed_hg_ECoG = extract_bandpower(processedECoG, clinicalFs, freq2use)
    % [b_hg,a_hg]=butter(4,[70 200]/(clinicalFs/2),'bandpass');
    % hg_ECoG=hilbert(filtfilt(b_hg,a_hg,processedECoG));
    % hg_ECoG = abs(hg_ECoG).^2;

    % firorder = floor((4/10) * clinicalFs);
    % delay = floor((firorder-1)/2);
    if strcmp(freq2use,'HG')
        bandpassFilter = [70 200]; % hz
    elseif strcmp(freq2use,'beta')
        bandpassFilter = [13 30];
    elseif strcmp(freq2use,'alpha')
        bandpassFilter = [8 12];
    elseif strcmp(freq2use,'theta')
        bandpassFilter = [4 8];
    end
    if ~strcmp(freq2use,'LMP')
        firorder = floor((4/10) * clinicalFs);
        [b6,a6] = fir1(firorder, bandpassFilter/(clinicalFs/2),'bandpass');
    else
        lmp_wsz = firorder;
        b6 = ones(lmp_wsz,1)/lmp_wsz;
        a6 = 1;
        % delay = floor((lmp_wsz-1)/2);
    end

    for channelnum = 1:size(processedECoG,2)
        highGammaBandSignal = filter(b6,a6, processedECoG(:,channelnum));
        % highGammaBandSignal = highGammaBandSignal(delay+1:end-delay);
        highGammaBandH = hilbert(highGammaBandSignal);
        hg_ECoG(:,channelnum) = abs(highGammaBandH).^2;
    end
    
    % Here we zscore the entire file since there is no baseline period
    % Do this downstream if not now
    % hg_ECoG = zscore(hg_ECoG, [], 1);

    % Smooth HG power
    % smoothed_hg_ECoG=zeros(size(hg_ECoG));
    % for i=1:size(hg_ECoG,2)
    %     smoothed=flip(movavg(flip(hg_ECoG(:,i)),"exponential",0.1*clinicalFs)); % causal 100ms moving exponential avg
    %     smoothed_hg_ECoG(:,i)=smoothed;
    % end
    smoothed_hg_ECoG = hg_ECoG;
end