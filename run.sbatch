#!/bin/bash
#SBATCH --job-name=gpu            # Название задачи
#SBATCH --error={task}-%j.err        # Файл для вывода ошибок
#SBATCH --output={task}-%j.log       # Файл для вывода результатов
#SBATCH --partition=gpu
#SBATCH --gpus=1                   # Требуемое кол-во GPU

./browanian params.txt histogram.data x.data y.data wx.data wy.data
