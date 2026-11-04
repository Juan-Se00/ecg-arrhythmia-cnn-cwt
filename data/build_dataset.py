"""End-to-end data pipeline: MIT-BIH → spectrogram PNGs.

Equivalent to `process_all_records()` from data_processing.py but
importable, parameterized, and side-effect free.
"""
import argparse
from pathlib import Path

from .load_mitbih import download_record, categorize_beat
from .remove_baseline import remove_baseline_wander
from .segment_beats import segment_beat
from ..features.generate_cwt import compute_spectrogram, save_spectrogram_png

DEFAULT_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124', '200', '201', '202', '203', '205', '207',
    '208', '209', '210', '212', '213', '214', '215', '217', '219', '220',
    '221', '222', '223', '228', '230', '231', '232', '233', '234',
]


def process_record(record_name: str, raw_dir: Path, out_dir: Path, degree: int = 6):
    record, annotation = download_record(record_name, raw_dir)
    fs = record.fs

    for lead_idx, lead_name in enumerate(['MLII', 'V5']):
        if lead_idx >= record.p_signal.shape[1]:
            continue
        signal = record.p_signal[:, lead_idx]

        for beat_idx, (peak, sym) in enumerate(
            zip(annotation.sample, annotation.symbol)
        ):
            beat = segment_beat(signal, peak, fs=fs)
            if beat is None:
                continue
            beat = remove_baseline_wander(beat, degree=degree)
            freqs, power_db = compute_spectrogram(beat, fs=fs)

            cls = categorize_beat(sym)
            out_path = out_dir / lead_name / cls / f'{record_name}_{beat_idx+1}.png'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_spectrogram_png(
                power_db, freqs, fs, len(beat), out_path,
                title=f'Record {record_name} | {lead_name} | {cls}',
            )


def main(args):
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = args.records or DEFAULT_RECORDS
    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] Processing record {rec}...")
        process_record(rec, raw_dir, out_dir, degree=args.degree)

    print(f"\nDone. Spectrograms → {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='data/raw')
    p.add_argument('--out-dir', default='data/processed/ECG_spectrograms')
    p.add_argument('--degree', type=int, default=6)
    p.add_argument('--records', nargs='*', default=None,
                   help='Subset of MIT-BIH record IDs (default: all 48)')
    main(p.parse_args())
