echo "2021-05-10 09:42"
python E:/FeatureAgeNet/train_step.py -lr 0.005 \
    -ep 100 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --outpath 'resnet34_step_coral_100' \
    -d 'resnet34 coral 标准方法 100轮学习率每30轮减一次'
