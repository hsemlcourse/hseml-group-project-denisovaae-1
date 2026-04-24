import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import requests
from settings import RAW_DATA_PATH


def save_from_url(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)

def resolve_kaggle_command():
    cli = shutil.which('kaggle')
    if cli is not None:
        return [cli]
    module_cmd = [sys.executable, '-m', 'kaggle.cli']
    probe = subprocess.run(module_cmd + ['--version'], check=False, capture_output=True, text=True)
    if probe.returncode == 0:
        return module_cmd
    return None

def save_from_kaggle(dataset, out_path):
    base_cmd = resolve_kaggle_command()
    if base_cmd is None:
        raise RuntimeError()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / '_tmp_kaggle'
    temp_dir.mkdir(parents=True, exist_ok=True)
    cmd = base_cmd + ['datasets', 'download', '-d', dataset, '-p', str(temp_dir), '--force']
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or 'unknown kaggle error'
        raise RuntimeError(stderr)
    zip_files = sorted(temp_dir.glob('*.zip'))
    if not zip_files:
        raise RuntimeError()
    zip_path = zip_files[-1]
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith('.csv')]
        if not csv_names:
            raise RuntimeError()
        target_name = out_path.name
        selected = target_name if target_name in csv_names else csv_names[0]
        zf.extract(selected, path=temp_dir)
    extracted = temp_dir / selected
    out_path.write_bytes(extracted.read_bytes())
    for file_path in temp_dir.glob('*'):
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path, ignore_errors=True)
    temp_dir.rmdir()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='')
    parser.add_argument('--dataset', type=str, default='saurabhshahane/electricity-load-forecasting')
    parser.add_argument('--out', type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()
    if args.url:
        save_from_url(args.url, args.out)
        print(f'Файл: {args.out}')
        return
    try:
        save_from_kaggle(args.dataset, args.out)
    except RuntimeError as err:raise SystemExit(1)
    print(f'Файл: {args.out}')


if __name__ == '__main__':
    main()
