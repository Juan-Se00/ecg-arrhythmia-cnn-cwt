"""Load MIT-BIH records via wfdb."""
from pathlib import Path
import wfdb

def download_record(record_name: str, database_dir: Path):
    wfdb.dl_database('mitdb', str(database_dir), records=[record_name])
    return (
        wfdb.rdrecord(f'{database_dir}/{record_name}'),
        wfdb.rdann(f'{database_dir}/{record_name}', 'atr'),
    )

def categorize_beat(symbol: str) -> str:
    BEAT_TYPES = {
        'N': 'Normal', 'j': 'Normal', 'J': 'Normal',
        'A': 'Supraventricular', 'a': 'Supraventricular', 'S': 'Supraventricular', 'e': 'Supraventricular',
        'V': 'Ventricular', 'r': 'Ventricular', 'R': 'Ventricular', 'E': 'Ventricular',
        'F': 'Fusion', 'f': 'Fusion',
        'Q': 'Unknown',
    }
    return BEAT_TYPES.get(symbol, 'Unknown')
