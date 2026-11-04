"""R-peak-centered 0.6 s beat segmentation (paper Eq. 1)."""
def segment_beat(signal, r_peak, fs=360, pre_s=0.2, post_s=0.4):
    pre  = int(pre_s * fs)
    post = int(post_s * fs)
    start, end = r_peak - pre, r_peak + post
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]
