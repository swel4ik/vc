## Description and Usage
```
    python copy_dirtree.py -s <path to source dir> -d <path to destination>
```
Copy TextGrid files to specified speakers directories
```
    python grid_to_dir.py '-s' <path to speakers directories> -g <path to grids directories>
```
Process LibriTTS structure (or similar) to WaveGrad 2 pipeline requirements
```
    python libritts_wavegrad2.py -s <path to speakers directories>
```
Rename txt files, remove original and alignment files from LibriTTS dataset
```
    python rename_txt.py -s <path to speakers directories>
```
Convert .TextGrid files to alignments.txt for synthesizer train (only for SV2TTS)
```
    python textgrid_to_alignments.py -s <path to speakers directories> -g <path to grids directories>
```