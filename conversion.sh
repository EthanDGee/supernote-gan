#!/bin/bash

find Note -name "*.note" -print0 | while read -d $'\0' file; do
  base=$(basename "${file}" .note)
  newFile="converted/${base}.png"
  echo "Converting ${base}"
  mkdir -p converted
  supernote-tool convert -t png -a "$file" "$newFile"
done
