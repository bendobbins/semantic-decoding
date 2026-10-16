import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)

import config
# from utils_ridge.interpdata import lanczosfun

# def get_lanczos_mat(oldtime, newtime, window = 3, cutoff_mult = 1.0, rectify = False):
#     """get matrix for downsampling from TR times to word times
#     """
#     cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult
#     sincmat = np.zeros((len(newtime), len(oldtime)))
#     for ndi in range(len(newtime)):
#         sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi] - oldtime, window)
#     return sincmat


def get_onset_mat(oldtime, newtime):
    width = np.mean(np.diff(newtime))
    first_edge = newtime[0] - width/2

    bininds = np.floor((np.asarray(oldtime) - first_edge + 1e-7) / width).astype(int)

    inrange = (bininds >= 0) & (bininds < len(newtime))

    onsetmat = np.zeros((len(newtime), len(oldtime)))
    onsetmat[bininds[inrange], np.nonzero(inrange)[0]] = 1

    return onsetmat

    
def affected_bins(start_index, end_index, onset_mat, delay = True):
    """identify TRs influenced by words in the range [start_index, end_index]
    """
    start_bin, end_bin = np.where(onset_mat[:, start_index])[0][0], np.where(onset_mat[:, end_index])[0][-1]
    start_bin, end_bin = start_bin + min(config.STIM_DELAYS), end_bin + max(config.STIM_DELAYS)
    start_bin, end_bin = max(start_bin, 0), min(end_bin, onset_mat.shape[0] - 1)
    return np.arange(start_bin, end_bin + 1)

    
class StimulusModel():
    """class for constructing stimulus features
    """
    def __init__(self, onset_mat, bin_stats, word_mean, device = 'cpu'):
        self.device = device
        self.onset_mat = torch.from_numpy(onset_mat).float().to(self.device)
        self.bin_mean = torch.from_numpy(bin_stats[0]).float().to(device)
        self.bin_std_inv = torch.from_numpy(np.diag(1 / bin_stats[1])).float().to(device)
        self.blank = torch.from_numpy(word_mean).float().to(self.device)
        self.word_bins = onset_mat.argmax(0)
        
    def _to_bins(self, bins, words, variants):
        """downsamples word embeddings to bin embeddings for each hypothesis
        """
        return torch.matmul(self.onset_mat[bins, words].unsqueeze(0), variants)
    
    def _normalize(self, variants):
        """normalize bin embeddings for each hypothesis
        """
        centered = variants - self.bin_mean
        return torch.matmul(centered, self.bin_std_inv)

    def _delay(self, variants, n_vars, n_feats):
        """apply finite impulse response delays to bin embeddings
        """
        delays = config.STIM_DELAYS
        n_bins = variants.shape[1]
        del_variants = torch.zeros(n_vars, n_bins, len(delays)*n_feats).to(self.device)
        for c, d in enumerate(delays): 
            feat_ind_start = c * n_feats
            feat_ind_end = (c + 1) * n_feats
            del_variants[:, d:, feat_ind_start:feat_ind_end] = variants[:, :n_bins - d, :]
        return del_variants
        
    def make_variants(self, sample_index, hypothesis_embs, var_embs, affected_bins):
        """create stimulus features for each hypothesis
        """
        n_variants, n_feats = len(var_embs), self.blank.shape[0]
        lo, hi = max(affected_bins[0] - max(config.STIM_DELAYS), 0), affected_bins[-1] + 1 # bins needed
        w0, w1 = np.searchsorted(self.word_bins, [lo, hi]) # words starting in bins [lo, hi)
        with torch.no_grad():
            full = self.blank.repeat(w1 - w0, 1) # window words x features
            full[:sample_index - w0] = torch.tensor(np.array(hypothesis_embs[w0:sample_index])).float().reshape(-1, n_feats).to(self.device)
            variants = full.repeat(n_variants, 1, 1) # variants x window words x features
            variants[:, sample_index - w0, :] = torch.tensor(np.array(var_embs)).float().to(self.device)
            bin_variants = self._normalize(self._to_bins(slice(lo, hi), slice(w0, w1), variants))
            del_bin_variants = self._delay(bin_variants, n_variants, n_feats)
        return del_bin_variants[:, affected_bins - lo, :].to('cpu')
            
class LMFeatures():
    """class for extracting contextualized features of stimulus words
    """
    def __init__(self, model, layer, context_words):
        self.model, self.layer, self.context_words = model, layer, context_words

    def extend(self, extensions, verbose = False):
        """outputs array of vectors corresponding to the last words of each extension
        """
        contexts = [extension[-(self.context_words+1):] for extension in extensions]
        if verbose: print(contexts)
        context_array = self.model.get_context_array(contexts)
        embs = self.model.get_hidden(context_array, layer = self.layer)
        return embs[:, len(contexts[0]) - 1]

    def make_stim(self, words):
        """outputs matrix of features corresponding to the stimulus words
        """
        context_array = self.model.get_story_array(words, self.context_words)
        embs = self.model.get_hidden(context_array, layer = self.layer)
        return np.vstack([embs[0, :self.context_words], 
            embs[:context_array.shape[0] - self.context_words, self.context_words]])