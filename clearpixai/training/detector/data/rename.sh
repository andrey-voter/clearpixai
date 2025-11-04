#!/bin/bash

DIR="clean"
PREFIX="clean"

cd "$DIR" || exit 1

i=0
for file in *.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    [ -f "$file" ] || continue
    ext="${file##*.}"
    printf -v newname "%s-%04d.%s" "$PREFIX" "$i" "$ext"
    mv "$file" "$newname"
    ((i++))
done

echo "✅ Renamed $i files in $DIR"
