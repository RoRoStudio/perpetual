# Before training
python -m features.export_parquet --overwrite


# With profiler
python -m models.train_unified.py --quick-test --profile

# Quick test (few minutes)
python -m models.train_unified.py --quick-test

# Full training (all instruments)
python -m models.train_unified.py

wandb agent robertfreericks-roro-studio/deribit-perpetual-model/24lt2kla --max-speed=true


# Run backtest
python -m simulate.backtest --model /mnt/p/perpetual/models/checkpoints/model_20250412_172238_best.pt


# Get resources details (bash command)
python -c "
import platform, psutil, GPUtil, shutil
print('\n🖥️ SYSTEM INFO\n' + '-'*30)
print(f'OS: {platform.system()} {platform.release()} ({platform.version()})')
print(f'Architecture: {platform.machine()}')
print(f'Python: {platform.python_version()}')
print(f'CPU: {platform.processor()}')
cpu_freq = psutil.cpu_freq()
print(f'Cores: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical')
print(f'CPU Frequency: {cpu_freq.current:.1f} MHz (Max: {cpu_freq.max:.1f} MHz)')
print(f'RAM: {psutil.virtual_memory().total / 1e9:.2f} GB total, {psutil.virtual_memory().available / 1e9:.2f} GB available')

print('\n🧠 GPU INFO\n' + '-'*30)
for gpu in GPUtil.getGPUs():
    print(f'{gpu.name}: {gpu.memoryTotal:.0f} MB total | {gpu.memoryUsed:.0f} MB used | {gpu.load*100:.1f}% load')

print('\n💽 DISK INFO\n' + '-'*30)
for part in psutil.disk_partitions():
    try:
        usage = shutil.disk_usage(part.mountpoint)
        print(f'{part.device} ({part.mountpoint}): {usage.used / 1e9:.2f} GB used | {usage.free / 1e9:.2f} GB free')
    except PermissionError:
        continue
"