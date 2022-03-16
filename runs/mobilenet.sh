echo "2021-11-25"
python E:/FeatureAgeNet/train.py -lr 0.001 \
    -ep 100 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --numworkers 2 \
    --outpath 'mobilenet_coral_100_no_3' \
    -d 'mobilenetV2 coral 标准方法 100轮学习率不减 优化了训练方式 使用了num_workers = 2 单输出模型'
