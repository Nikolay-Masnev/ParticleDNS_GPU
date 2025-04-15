#!/bin/bash
#SBATCH --job-name=gpu            # Название задачи
#SBATCH --error={task}-%j.err        # Файл для вывода ошибок
#SBATCH --output={task}-%j.log       # Файл для вывода результатов
#SBATCH --partition=gpu
#SBATCH --gpus=1                   # Требуемое кол-во GPU

./browanian_tc_1000 params.txt histogram_1000.data x.data y.data wx.data wy.data
