#!/bin/bash

read -p "Download New Model？[yes/no]: " download_choice

if [ "$download_choice" = "yes" ]; then
  read -p "Input model name like'TheBloke/Xwin-LM-13B-v0.2-GPTQ': " model_name
  python download-model.py "$model_name" --output ./models && echo "Model downloaded !"
fi

echo "Models you can choose："
i=1
for dir in models/*; do
  if [ -d "$dir" ]; then
    echo "$i> $(basename "$dir")"
    ((i++))
  fi
done

read -p "Please select one(use number in list): " choice

selected_dir=$(ls -d models/*/ | sed -n "${choice}p" | xargs basename)
sed -i "s/model_name: .*/model_name: $selected_dir/" config.yml

echo "Chat templates you can choose："
i=1
for file in templates/*.jinja; do
  if [ -f "$file" ]; then
    echo "$i> $(basename -s .jinja "$file")"
    ((i++))
  fi
done
read -p "Please select one(use number in list): " file_choice

selected_file=$(ls templates/*.jinja | sed -n "${file_choice}p" | xargs basename -s .jinja)
sed -i "s/prompt_template: .*/prompt_template: $selected_file/" config.yml


read -p "Input api port: " port_number
sed -i "s/port: .*/port: $port_number/" config.yml


read -p "Input max_seq_len: " max_seq_len_number
sed -i "s/max_seq_len: .*/max_seq_len: $max_seq_len_number/" config.yml

read -p "Automatically assign GPU? [yes/no]: " gpu_choice

if [ "$gpu_choice" = "yes" ]; then
  sed -i "s/gpu_split_auto: .*/gpu_split_auto: True/" config.yml
else
  sed -i "s/gpu_split_auto: .*/gpu_split_auto: False/" config.yml
  read -p "Please input the GPU divide array, for example: 10,20: " gpu_divide_array
  sed -i "s/gpu_split: .*/gpu_split: [$gpu_divide_array]/" config.yml
fi

python main.py
