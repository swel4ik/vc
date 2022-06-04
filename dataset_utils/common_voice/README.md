## Description and Usage
Preprocess common voice Dataset

Create directories according to tsv file
```
    python create_dirs.py -p <path to tsv file> -d <destination folder> 
```
Move files to required speakers directories, excluding speakers with less than 4 samples
```
    python filter_counts.py  -p <path to tsv file> -d <destination folder> -w <path to wavs files>
```
Estimate number of speakers with samples count less than threshold
```
    python speakers_estimation.py -s <path to speakers directory> -t <samples threshold>
```
Convert mp3 to wav files
```
    sh mp3_to_wav.sh
```