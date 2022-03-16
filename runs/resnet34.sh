echo "2021-12-02"
python E:/FeatureAgeNet/train.py -lr 0.0005 \
    -ep 200 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --numworkers 2 \
    --SE \
    --outpath 'resnet_coral_200_SE_20211202' \
    -d 'resnet34 coral 输入64核 标准方法 200轮学习率不减 优化了训练方式 使用了num_workers = 2 单输出模型 SE module'
