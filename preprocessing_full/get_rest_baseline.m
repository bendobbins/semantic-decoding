function baseline_signal = get_rest_baseline(data, cfg)
% Extract inter-trial baseline segments (no concatenation).
%
% Output: baseline_signal (struct array)
%   .trial_idx      -> trial number the baseline precedes (i+1)
%   .start_s        -> start time (sec, absolute)
%   .end_s          -> end time (sec, absolute)
%   .start_idx      -> start sample (absolute index)
%   .end_idx        -> end sample (absolute index)
%   .signal         -> [T x n_channels] baseline bandpower segment

    Fs = data.clinicalFs;
    all_data = data.all_data;
    n_trials = size(all_data,1);

    % Constants
    buffer_post = cfg.baseline.buffer_post; % sec after voice offset
    buffer_pre  = cfg.baseline.buffer_pre;  % sec before next trial
    min_dur     = cfg.baseline.min_dur;     % minimum ITI duration (sec)

    baseline_signal = struct( ...
        'trial_idx', {}, ...
        'start_s', {}, ...
        'end_s', {}, ...
        'start_idx', {}, ...
        'end_idx', {}, ...
        'signal', {} ...
    );

    bcount = 0;

    % Exclude first and last trials by construction
    for i = 1:(n_trials-1)
        % Require both trials to be good
        if ~all_data{i,10} || ~all_data{i+1,10}
            continue
        end

        % Require valid voice offset for trial i
        if isempty(data.voice_offset(i)) || ~isfinite(data.voice_offset(i))
            continue
        end

        % Absolute times (seconds)
        t_start = data.voice_offset(i) + buffer_post;
        t_end   = data.trial_start(i+1) - buffer_pre;
        fprintf("Trial %d: %.2f - %.2f (%.2f)\n",i,t_start,t_end,t_end-t_start)

        % Duration check
        if (t_end - t_start) < min_dur
            continue
        end

        % Convert to sample indices
        idx_start = floor(t_start * Fs) + 1;
        idx_end   = floor(t_end   * Fs);

        % Bounds check
        if idx_start < 1 || idx_end > size(data.ECoGDataMatrix_final,1)
            continue
        end

        % Extract baseline segment from *bandpower signal*
        % NOTE: assumes this function is called after bandpower computation
        segment = data.bandpower(idx_start:idx_end, :);

        % Store
        bcount = bcount + 1;
        baseline_signal(bcount).trial_idx = i + 1;
        baseline_signal(bcount).start_s   = t_start;
        baseline_signal(bcount).end_s     = t_end;
        baseline_signal(bcount).start_idx = idx_start;
        baseline_signal(bcount).end_idx   = idx_end;
        baseline_signal(bcount).signal    = segment;
    end

    plot_baseline_segments(data, baseline_signal);
end
