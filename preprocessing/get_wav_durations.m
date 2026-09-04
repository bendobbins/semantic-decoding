function stim_info = get_wav_durations(stims_dir, target_labels, varargin)
% GET_WAV_DURATIONS  Reads .wav files from a directory, computes durations,
% and transcribes them with Whisper (word-level onsets/offsets).
%
% Usage:
%   stim_info = get_wav_durations(stims_dir, target_labels, ...
%                 'CondaEnv','Speech','Model','small','Language','en');
%
% Returns a dictionary where each key = label, and each value is a struct:
%   .duration_s      : audio duration (s)
%   .stim_onset_s    : detected acoustic onset (leading silence)
%   .transcript      : full text transcription
%   .words           : table [word, onset_s, offset_s] (absolute times)

% ---------------- Parse optional params ----------------
p = inputParser;
addParameter(p,'CondaEnv',"Speech",@(s)isstring(s)||ischar(s));
addParameter(p,'Model',"small",@(s)isstring(s)||ischar(s));
addParameter(p,'Language',"en",@(s)isstring(s)||ischar(s));
addParameter(p,'OutputDir',string(tempdir),@(s)isstring(s)||ischar(s));
addParameter(p,'FP16',false,@(b)islogical(b)||isnumeric(b));
addParameter(p,'AlignToAudioStart',true,@(b)islogical(b)||isnumeric(b));
% addParameter(p,'UseSavedOutputs',false,@(b)islogical(b)||isnumeric(b));
parse(p,varargin{:});


condaEnv = string(p.Results.CondaEnv);
model    = string(p.Results.Model);
lang     = string(p.Results.Language);
outDir   = string(p.Results.OutputDir);
useFP16  = logical(p.Results.FP16);
alignAbs = logical(p.Results.AlignToAudioStart);
% useSavedOutputs = logical(p.Results.UseSavedOutputs);

% ---------------- Check directory ----------------
if ~isfolder(stims_dir)
    error('Directory does not exist: %s', stims_dir);
end

% ---------------- Initialize dictionary ----------------
stim_info = dictionary;

% ---------------- Loop over labels ----------------
% Get all .wav files once
files = dir(fullfile(stims_dir, "*.wav"));
all_names = string({files.name});

% Create lowercase lookup for case-insensitive matching
all_names_lower = lower(all_names);

for i = 1:numel(target_labels)
    label = string(target_labels{i});
    target_name = label + ".wav";
    
    % Case-insensitive match
    idx = find(all_names_lower == lower(target_name), 1);
    
    if isempty(idx)
        warning('File not found (any case variant): %s', target_name);
        continue;
    end
    
    % Use the first matching file
    filename = all_names(idx);
    filepath = fullfile(stims_dir, filename);
    
    % fprintf('Using file: %s\n', filepath);

    % ---------- Duration ----------
    info = audioinfo(filepath);
    duration = info.Duration;

    % ---------- Acoustic onset (leading silence) ----------
    % stim_onset_s = detect_leading_silence(filepath);
    stim_onset_s = 0;

    % ---------- Transcription with Whisper ----------
    try
        [transcript, words] = run_whisper(filepath, ...
            'CondaEnv',condaEnv,'Model',model,'Language',lang, ...
            'OutputDir',outDir,'FP16',useFP16,'AlignToAudioStart',alignAbs, ...
            'SilenceOffset',stim_onset_s);
    catch ME
        warning('Whisper failed for %s: %s', label, ME.message);
        transcript = "";
        words = table();
    end

    % ---------- Store results ----------
    stim_info(label) = struct( ...
        'duration_s',duration, ...
        'stim_onset_s',stim_onset_s, ...
        'transcript',transcript, ...
        'words',words);

    fprintf('Processed %-20s | %.2fs | %s\n', label, duration, transcript);
end

fprintf('\nProcessed %d .wav files total.\n', numEntries(stim_info));
end

% =====================================================================
% =============== HELPER FUNCTIONS BELOW ==============================
% =====================================================================

function [transcript, words] = run_whisper(filepath, varargin)
% Wrapper around Whisper CLI to produce word-level timestamps
p = inputParser;
addParameter(p,'CondaEnv',"Speech");
addParameter(p,'Model',"small");
addParameter(p,'Language',"en");
addParameter(p,'OutputDir',tempdir);
addParameter(p,'FP16',false);
addParameter(p,'AlignToAudioStart',true);
addParameter(p,'SilenceOffset',0);
% addParameter(p,'UseSavedOutputs',false);
parse(p,varargin{:});

condaEnv = string(p.Results.CondaEnv);
model    = string(p.Results.Model);
lang     = string(p.Results.Language);
outDir   = string(p.Results.OutputDir);
useFP16  = logical(p.Results.FP16);
alignAbs = logical(p.Results.AlignToAudioStart);
offset   = double(p.Results.SilenceOffset);
% useSavedOutputs = logical(p.Results.UseSavedOutputs);

[~,base,~] = fileparts(filepath);
base = lower(base);

% if ~useSavedOutputs
% Check if output has already been saved
jsonPath = find_whisper_json(outDir, base);
if jsonPath == ""
    fprintf("No existing transcribed output found. Running Whisper transcription...\n")
    args = " --model " + model + ...
           " --word_timestamps True --output_format json" + ...
           " --output_dir " + shquote(outDir);
    if strlength(lang)>0
        args = args + " --language " + lang;
    end
    if ~useFP16
        args = args + " --fp16 False";
    end
    
    audioArg = shquote(filepath);
    if strlength(condaEnv)>0
        cmd = "conda run -n " + condaEnv + " whisper " + audioArg + args;
    else
        cmd = "whisper " + audioArg + args;
    end
    
    [st,msg] = system(cmd);
    if st ~= 0
        error("Whisper error:\n%s", msg);
    end
else
    fprintf("Using existing transcription found at %s\n", jsonPath)
end

jsonPath = find_whisper_json(outDir, base);
if jsonPath == ""
    jsonPath = find_whisper_json(pwd, base);
end
if jsonPath == ""
    error("No JSON output found for %s", base);
end

raw = fileread(jsonPath);
J = jsondecode(raw);

transcript = "";
if isfield(J,'text'); transcript = string(J.text); end

% Extract words
words = collect_words_from_json(J);
if alignAbs && offset > 0 && ~isempty(words)
    words.onset_s  = words.onset_s  + offset;
    words.offset_s = words.offset_s + offset;
end
end

function t0 = detect_leading_silence(wavPath)
% Simple RMS-based leading silence detector
[x,fs] = audioread(wavPath);
if size(x,2)>1, x = mean(x,2); end
x = x - mean(x);
Nf = round(0.02*fs);
Nh = round(0.01*fs);
numFrames = floor((numel(x)-Nf)/Nh);
rmsVals = zeros(numFrames,1);
for i=1:numFrames
    idx = (1:Nf)+(i-1)*Nh;
    rmsVals(i) = sqrt(mean(x(idx).^2));
end
tFrames = ((0:numFrames-1)'*Nh)/fs;
thr = max(0.01, median(rmsVals)*5);
idx = find(rmsVals>thr,1);
if isempty(idx)
    t0 = 0;
else
    t0 = tFrames(idx);
end
end

function T= collect_words_from_json(J)
words = strings(0,1); onsets = []; offsets = [];
if isfield(J,'segments')
    for s = 1:numel(J.segments)
        seg = J.segments(s);
        if isfield(seg,'words')
            for w = 1:numel(seg.words)
                ww = seg.words(w);
                if isfield(ww,'word')
                    tok = strtrim(string(ww.word));
                    if tok ~= ""
                        words(end+1,1)=tok;
                        onsets(end+1,1)=double(ww.start);
                        offsets(end+1,1)=double(ww.end);
                    end
                end
            end
        end
    end
end
T = table(words,onsets,offsets,'VariableNames',{'word','onset_s','offset_s'});
end

function jsonPath = find_whisper_json(outDir, base)
% handle various naming quirks
cands = [
    fullfile(outDir, base+".json")
    fullfile(outDir, base+".wav.json")
    fullfile(outDir, base+".en.json")
];
for i = 1:numel(cands)
    if exist(cands(i),'file'), jsonPath = cands(i); return; end
end
D = dir(fullfile(outDir, base+"*.json"));
if ~isempty(D)
    jsonPath = string(fullfile(D(1).folder,D(1).name));
else
    jsonPath = "";
end
end

function s = shquote(x)
if ispc
    s = '"' + string(x) + '"';
else
    s = "'" + string(x) + "'";
end
end