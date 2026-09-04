% Main preprocessing_full function
function preprocess(cfg)
    data = prepare_data(cfg);
    try
        if cfg.flags.inspectBeforeSaving
            data = inspect_data(data);
        end
    catch
    end
    save_data_by_freq(data, cfg);
end