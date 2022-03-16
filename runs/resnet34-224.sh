echo "2021-12-07"
python E:/FeatureAgeNet/train_224.py -lr 0.0005 \
    -ep 90 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --model 'mobilenet' \
    --img_size 224 \
    --numworkers 0 \
    --outpath 'mobilenetv3_rank_224_224_size_90_epoches' \
    -d 'mobilenetv3 rank 输入64核 标准方法 90轮学习率不减 优化了训练方式 使用了num_workers = 2 单输出模型'
