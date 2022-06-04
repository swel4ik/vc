#!/bin/bash
for i in `find . -depth -type d`
do 
for f in $i/*.mp3
do ffmpeg -i "$f" -acodec pcm_s16le -ac 1 -ar 22050 "${f%.mp3}.wav"
rm $f
done

done
