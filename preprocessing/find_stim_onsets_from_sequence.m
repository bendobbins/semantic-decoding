function [onsets, classes, map] = find_stim_onsets_from_sequence(sequence, stimuliValues, opts)
% FIND_STIM_ONSETS_FROM_SEQUENCE
% Build SAME-LENGTH, row-aligned starts/ends per trial from sequence + Stimuli.Values.
%
% StimuliValues rows:
%   row 1 -> text
%   row 2 -> text OR picture (decide by extension)
%   row 3 -> audio (highest priority)
% If more rows exist, they're ignored for typing (audio still wins if present).
%
% Outputs (inclusive ends):
%   onsets.stim, onsets.stim_end
%   onsets.fixation, onsets.fixation_end     % merged across consecutive fixation-class blocks
%   onsets.go, onsets.go_end                 % merged across consecutive go-class blocks
%   onsets.codes_stim / onsets.codes_fixation / onsets.codes_go   (code of the FIRST block picked)
%
%   classes.stimulus_codes / fixation_codes / go_codes
%   map.labels : 1..N lowercased labels (base names for file paths, or lowercased text)
%   map.code2lab : containers.Map(code -> label)
%   map.types  : 1..N {'audio'|'picture'|'text'|''}  (content type per code)
%
% Options (all optional):
%   .fix_patterns       : default {'fix','fixation','cross','plus','center','baseline','rest','blank'}
%   .go_patterns        : default {'go','go_cue','go-cue','cue','beep','ding','readygo','start','green'}
%   .ignore_codes       : default [0]
%   .explicit.fix_codes : numeric vector (force fixation)
%   .explicit.go_codes  : numeric vector (force go)
%   .drop_incomplete    : default false (drop rows that miss any of stim/fix/go)
%
if nargin < 3, opts = struct(); end
fix_patterns    = getOpt(opts,'fix_patterns', {'fix','fixation','cross','plus','center','baseline','rest','blank'});
go_patterns     = getOpt(opts,'go_patterns',  {'go_cue','go-cue','cue','beep','ding','readygo','start','green'});
ignore_codes    = getOpt(opts,'ignore_codes',[0]);
drop_incomplete = getOpt(opts,'drop_incomplete', false);

% --- Labels & CONTENT TYPES -------------------------------------------------
[labels, labels_raw, types] = labels_from_stimuli_values(stimuliValues); %#ok<NASGU>
N = numel(labels);

% --- Classify codes into fixation / go by NAME KEYWORDS (stimulus = everything else)
isFix = false(1,N); isGo = false(1,N);
for i = 1:N
    lab = labels{i};
    if ~isempty(lab)
        isFix(i) = any(cellfun(@(p) ~isempty(regexpi(lab,p)), fix_patterns));
        isGo(i)  = any(cellfun(@(p) ~isempty(regexpi(lab,p)), go_patterns));
    else
        isFix(i) = true;
    end
end
if isfield(opts,'explicit')
    if isfield(opts.explicit,'fix_codes') && ~isempty(opts.explicit.fix_codes)
        isFix(:) = false; isFix(unique_clip(opts.explicit.fix_codes,N)) = true;
    end
    if isfield(opts.explicit,'go_codes')  && ~isempty(opts.explicit.go_codes)
        isGo(:) = false;  isGo(unique_clip(opts.explicit.go_codes, N))  = true;
    end
end
fixation_codes = find(isFix);
go_codes       = find(isGo);
stimulus_codes = setdiff(1:N, union(fixation_codes, go_codes));   % all remaining are "stimuli"

% --- Contiguous blocks from SEQUENCE ---------------------------------------
seq = double(sequence(:));
if isempty(seq), error('find_stim_onsets_from_sequence:EmptySequence','sequence is empty.'); end
T = numel(seq);
starts = [1; find(diff(seq)~=0)+1];
ends   = [starts(2:end)-1; T];
codes  = seq(starts);

% keep valid / non-ignored blocks
keep   = isfinite(codes) & (codes>=1 & codes<=N) & ~ismember(codes, ignore_codes);
starts = starts(keep); ends = ends(keep); codes = codes(keep);

% per-block class index: 1=stimulus, 2=fixation, 3=go
cls = zeros(size(codes));
cls(ismember(codes, stimulus_codes)) = 1;
cls(ismember(codes, fixation_codes)) = 2;
cls(ismember(codes, go_codes))       = 3;

% --- Pair trials: STIMULUS-anchored; find FIRST fixation and FIRST go before next stimulus
stim_idx = find(cls==1);
nS = numel(stim_idx);

stim_start = starts(stim_idx);   stim_end = ends(stim_idx);   stim_code = codes(stim_idx);
fix_start  = nan(nS,1);          fix_end  = nan(nS,1);        fix_code  = nan(nS,1);
go_start   = nan(nS,1);          go_end   = nan(nS,1);        go_code   = nan(nS,1);

M = numel(cls);
for k = 1:nS
    i0 = stim_idx(k);
    if k < nS
        i1 = stim_idx(k+1);          % next stimulus block index
        sr = (i0+1):(i1-1);          % search range inside this trial
    else
        sr = (i0+1):M;               % last stimulus -> search to the end
    end
    if isempty(sr), continue; end

    % ---- FIXATION: first fixation in sr, then MERGE across consecutive fixation-class blocks
    ix_fix = find(cls(sr)==2, 1, 'first');
    if ~isempty(ix_fix)
        j  = sr(ix_fix);             % first fixation block index
        fix_start(k) = starts(j);
        j2 = j; upper = sr(end);
        while j2 < upper && cls(j2+1) == 2
            j2 = j2 + 1;
        end
        fix_end(k)  = ends(j2);      % INCLUSIVE end of merged fixation
        fix_code(k) = codes(j);      % keep the code of the FIRST block
    end

    % ---- GO: first go in sr, then MERGE across consecutive go-class blocks
    ix_go = find(cls(sr)==3, 1, 'first');
    if ~isempty(ix_go)
        j  = sr(ix_go);              % first go block index
        go_start(k) = starts(j);
        j2 = j; upper = sr(end);
        while j2 < upper && cls(j2+1) == 3
            j2 = j2 + 1;
        end
        go_end(k)  = ends(j2);       % INCLUSIVE end of merged go (TRIAL END by your def)
        go_code(k) = codes(j);       % keep the code of the FIRST block
    end
end

% Optional: drop incomplete rows
if drop_incomplete
    keepRows = ~isnan(stim_start) & ~isnan(fix_start) & ~isnan(go_start) & ...
               ~isnan(stim_end)   & ~isnan(fix_end)   & ~isnan(go_end);
    stim_start = stim_start(keepRows); stim_end = stim_end(keepRows); stim_code = stim_code(keepRows);
    fix_start  = fix_start(keepRows);  fix_end  = fix_end(keepRows);  fix_code  = fix_code(keepRows);
    go_start   = go_start(keepRows);   go_end   = go_end(keepRows);   go_code   = go_code(keepRows);
end

% --- Outputs ---------------------------------------------------------------
onsets.stim         = stim_start(:);
onsets.stim_end     = stim_end(:);
onsets.fixation     = fix_start(:);
onsets.fixation_end = fix_end(:);     % merged, inclusive
onsets.go           = go_start(:);
onsets.go_end       = go_end(:);      % merged, inclusive

onsets.codes_stim      = stim_code(:);
onsets.codes_fixation  = fix_code(:);
onsets.codes_go        = go_code(:);

% if the first itemt in onsets.stim is 1. remove that item in all fields in
% onsets
if onsets.stim(1)==1
    onsets = structfun(@(x) x(2:end), onsets, 'UniformOutput', false);
end

classes.stimulus_codes  = stimulus_codes;
classes.fixation_codes  = fixation_codes;
classes.go_codes        = go_codes;

map.labels   = labels;
map.code2lab = containers.Map('KeyType','double','ValueType','char');
for i = 1:numel(labels), map.code2lab(i) = labels{i}; end
map.types    = types(:)';     % 'audio' | 'picture' | 'text' | ''

end % main


% ======================= Helpers =======================

function v = getOpt(s,f,def), v = def; if isstruct(s) && isfield(s,f), v = s.(f); end, end

function idx = unique_clip(v, N)
    v = v(:)'; v = v(isfinite(v) & v>=1 & v<=N);
    idx = unique(round(v));
end

function [labels, labels_raw, types] = labels_from_stimuli_values(StimuliValues)
% Build per-code label + raw string + CONTENT TYPE:
%   type = 'audio'   if row3 non-empty
%   type = 'picture' if row2 has an image extension
%   type = 'text'    if row2 has no image ext, else row1 text
%   type = ''        if nothing present
    labels = {}; labels_raw = {}; types = {};
    if isstruct(StimuliValues) && isfield(StimuliValues,'Values')
        StimuliValues = StimuliValues.Values;
    end

    if ~iscell(StimuliValues)
        % Fallback: a flat list (lines or '|' separated)
        parts = regexp(char(string(StimuliValues)), '\r?\n|\|', 'split');
        parts = parts(~cellfun(@isempty, strtrim(parts)));
        N = numel(parts);
        labels = cell(1,N); labels_raw = cell(1,N); types = cell(1,N);
        for i = 1:N
            t = strip_quotes_idx(parts{i});
            labels_raw{i} = t;
            [~, base, ext] = fileparts(t);
            if ~isempty(ext) && isImageExt(ext), types{i} = 'picture'; else, types{i} = 'text'; end
            labels{i} = lower(ifempty(base, t));
        end
        return;
    end

    nRows = size(StimuliValues,1);
    nCols = size(StimuliValues,2);
    labels     = cell(1,nCols);
    labels_raw = cell(1,nCols);
    types      = cell(1,nCols);

    row1 = []; row2 = []; row3 = [];
    if nRows >= 1, row1 = StimuliValues(1,:); end   % text
    if nRows >= 2, row2 = StimuliValues(2,:); end   % text OR picture
    if nRows >= 3, row3 = StimuliValues(3,:); end   % audio

    for i = 1:nCols
        s1 = ''; s2 = ''; s3 = '';
        if ~isempty(row1), s1 = strip_quotes_idx(char(string(row1{i}))); end
        if ~isempty(row2), s2 = strip_quotes_idx(char(string(row2{i}))); end
        if ~isempty(row3), s3 = strip_quotes_idx(char(string(row3{i}))); end

        if ~isempty(s3)                               % AUDIO wins
            chosen = s3; types{i} = 'audio';
            [~, base, ~] = fileparts(chosen);
            labels{i} = lower(base); labels_raw{i} = chosen;

        elseif ~isempty(s2)                           % Row 2: picture or text
            [parent, base, ext] = fileparts(s2);
            [~, parent_base, ~] = fileparts(parent);
            if (~isempty(parent_base) && contains(parent_base, "_words"))
            % if (~isempty(ext) && ext==".png")
                types{i} = 'text';
                labels{i} = lower(ifempty(base, s2)); labels_raw{i} = s2;
            elseif ~isempty(ext) && isImageExt(ext)
                types{i} = 'picture';
                labels{i} = lower(base); labels_raw{i} = s2;
            else
                types{i} = 'text';
                labels{i} = lower(ifempty(base, s2)); labels_raw{i} = s2;
            end

        elseif ~isempty(s1)                           % Row 1: text
            types{i} = 'text';
            [~, base, ~] = fileparts(s1);
            labels{i} = lower(ifempty(base, s1)); labels_raw{i} = s1;

        else                                          % empty column
            types{i}  = '';
            labels{i} = '';
            labels_raw{i} = '';
        end
    end
end

function tf = isImageExt(ext)
    ext = lower(string(ext));
    tf = ismember(ext, [".png",".jpg",".jpeg",".bmp",".gif",".tif",".tiff",".svg",".webp"]);
end

function s = strip_quotes_idx(s)
    % Trim, drop leading "NN:" or "NN =" index, strip quotes, normalize slashes.
    s = strtrim(s);
    m = regexp(s, '^\s*(\d+)\s*[:=]\s*(.*)$', 'tokens', 'once');
    if ~isempty(m), s = m{2}; end
    s = regexprep(s, '^[\s''"]+|[\s''"]+$', '');
    s = strrep(s, '\', '/');         % normalize path separators
    s = regexprep(s, '/+', '/');     % collapse multiple slashes
    if numel(s) > 1, s = regexprep(s, '(?<!^)/+$', ''); end
end

function y = ifempty(a, b)
    if isempty(a), y = b; else, y = a; end
end