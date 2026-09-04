function plot_baseline_segments(data, baseline_signal)
% Diagnostic visualization for extracted baseline segments
    trial_start  = data.trial_start;
    voice_onset  = data.voice_onset;
    voice_offset = data.voice_offset;

    n_segments = length(baseline_signal);

    %% Print summary
    fprintf("\n=== Baseline extraction summary ===\n");
    fprintf("Number of baseline segments: %d\n", n_segments);

    total_dur = 0;
    for k = 1:n_segments
        dur = baseline_signal(k).end_s - baseline_signal(k).start_s;
        total_dur = total_dur + dur;
        fprintf("  Segment %2d | precedes trial %3d | duration = %.3f s\n", ...
            k, baseline_signal(k).trial_idx, dur);
    end
    fprintf("Total baseline duration: %.3f s\n", total_dur);

    %% Plot
    figure; hold on;
    ylims = [0 1];

    % Trial onsets
    for i = 1:length(trial_start)
        xline(trial_start(i), 'k-', 'LineWidth', 1);
    end

    % Voice onset
    for i = 1:length(voice_onset)
        if isfinite(voice_onset(i))
            xline(voice_onset(i), 'b--', 'LineWidth', 1);
        end
    end

    % Voice offset
    for i = 1:length(voice_offset)
        if isfinite(voice_offset(i))
            xline(voice_offset(i), 'r--', 'LineWidth', 1);
        end
    end

    % Baseline segments (shaded)
    for k = 1:n_segments
        xs = baseline_signal(k).start_s;
        xe = baseline_signal(k).end_s;

        patch( ...
            [xs xe xe xs], ...
            [ylims(1) ylims(1) ylims(2) ylims(2)], ...
            [0.6 0.9 0.6], ...
            'FaceAlpha', 0.4, ...
            'EdgeColor', 'none' ...
        );
    end

    ylim(ylims);
    xlabel('Time (s)');
    yticks([]);
    title('Baseline segments (green) with event markers');

    legend_entries = { ...
        'Trial onset', ...
        'Voice onset', ...
        'Voice offset', ...
        'Baseline segment' ...
    };

    % Dummy handles for legend consistency
    h_trial  = plot(nan, nan, 'k-');
    h_vo     = plot(nan, nan, 'b--');
    h_vf     = plot(nan, nan, 'r--');
    h_base   = patch(nan, nan, [0.6 0.9 0.6], 'FaceAlpha', 0.4);

    % legend([h_trial h_vo h_vf h_base], legend_entries, 'Location', 'best');

    hold off;
end