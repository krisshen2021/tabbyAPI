#!/bin/bash
source venv/bin/activate
echo "TabbyAPI Startup Script - Muffin"
read -p "Run last configuration? [y/n]: " runlast
if [ "$runlast" = "y" ]; then
 python main.py
elif [ "$runlast" = "n" ]; then
 read -p "Disable Authorization? [y/n]: " disableAuth
 if [ "$disableAuth" = "y" ]; then
   sed -i "s/disable_auth: .*/disable_auth: True/" config.yml
 elif [ "$disableAuth" = "n" ]; then
   sed -i "s/disable_auth: .*/disable_auth: False/" config.yml
 else
   echo "Invalid input. Exiting."
   exit 1
 fi

 read -p "Download New Model? [y/n]: " download_choice

 if [ "$download_choice" = "y" ]; then
   read -p "Input model name like'TheBloke/Xwin-LM-13B-v0.2-GPTQ': " model_name
   python download-model.py "$model_name" --output ./models && echo "Model downloaded !"
 fi

 echo "Models you can choose: "
 model_options=()
 i=1
 for dir in models/*; do
   if [ -d "$dir" ]; then
     echo "$i> $(basename "$dir")"
     model_options+=($i)
     ((i++))
   fi
 done

 read -p "Please select one(use number in list): " choice
 if [[ ! " ${model_options[@]} " =~ " ${choice} " ]]; then
     echo "Invalid input. Exiting."
     exit 1
 fi

 selected_dir=$(ls -d models/*/ | sed -n "${choice}p" | xargs basename)
 sed -i "s/model_name: .*/model_name: $selected_dir/" config.yml

 read -p "Using Chat Complatetion? [y/n]: " useChatComp
   if [ "$useChatComp" = "n" ]; then
     sed -i "s/prompt_template: .*/prompt_template: None/" config.yml
   elif [ "$useChatComp" = "y" ]; then
     echo "Chat templates you can choose: "
     template_options=()
     i=1
     for file in templates/*.jinja; do
       if [ -f "$file" ]; then
         echo "$i> $(basename -s .jinja "$file")"
         template_options+=($i)
         ((i++))
       fi
     done
     read -p "Please select one(use number in list): " file_choice
     if [[ ! " ${template_options[@]} " =~ " ${file_choice} " ]]; then
     echo "Invalid input. Exiting."
     exit 1
     fi
     selected_file=$(ls templates/*.jinja | sed -n "${file_choice}p" | xargs basename -s .jinja)
     sed -i "s/prompt_template: .*/prompt_template: $selected_file/" config.yml
   else
     echo "Invalid input. Exiting."
     exit 1
   fi

 read -p "Input api port: " port_number
 if [[ $port_number =~ ^[0-9]+$ ]]; then
     sed -i "s/port: .*/port: $port_number/" config.yml
 else
     echo "Invalid input. Exiting."
     exit 1
 fi

 read -p "Input max_seq_len: " max_seq_len_number
 if [[ $max_seq_len_number =~ ^[0-9]+$ ]]; then
     sed -i "s/max_seq_len: .*/max_seq_len: $max_seq_len_number/" config.yml
 else
     echo "Invalid input. Exiting."
     exit 1
 fi

 read -p "Automatically assign GPU? [y/n]: " gpu_choice

 if [ "$gpu_choice" = "y" ]; then
   sed -i "s/gpu_split_auto: .*/gpu_split_auto: True/" config.yml
 elif [ "$gpu_choice" = "n" ]; then
   sed -i "s/gpu_split_auto: .*/gpu_split_auto: False/" config.yml
   read -p "Please input the GPU divide array, for example: 10,20: " gpu_divide_array
   sed -i "s/gpu_split: .*/gpu_split: [$gpu_divide_array]/" config.yml
 else
   echo "Invalid input. Exiting."
   exit 1
 fi

 python main.py

fi
