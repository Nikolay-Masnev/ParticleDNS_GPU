#!/bin/bash
#SBATCH --job-name=gpu            # Название задачи
#SBATCH --error=%j.err        # Файл для вывода ошибок
#SBATCH --output=%j.log       # Файл для вывода результатов
#SBATCH --partition=gpu
#SBATCH --gpus=1                   # Требуемое кол-во GPU

./browanian_zero_corr params.txt histogram_zero_corr.data x.data y.data wx.data wy.data hist_2D.data vel_var.data
