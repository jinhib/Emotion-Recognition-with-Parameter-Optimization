import shutil
import os

base_dir = 'C:/Users/HP/Desktop/배진희'

ori_dir = base_dir + '/' + 'EX_ORG_labeling'

for file_num in range(0, 7):

    copy_train_dir = base_dir + '/' + 'EX_ORG/train/6'
    copy_test_dir = base_dir + '/' + 'EX_ORG/test/6'
    file_list = os.listdir(ori_dir)
    idx = int(len(file_list)*0.7)
    train_list = file_list[:idx]
    test_list = file_list[idx:]

    for i in train_list:
        shutil.copy(ori_dir+'/'+ i, copy_train_dir)

    for i in test_list:
        shutil.copy(ori_dir + '/' + i, copy_test_dir)