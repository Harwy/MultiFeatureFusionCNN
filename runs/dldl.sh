echo "2021-12-03"
python E:/FeatureAgeNet/train.py -lr 0.0005 \
    -ep 200 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'dldl' \
    --img_size 120 \
    --numworkers 2 \
    --outpath 'resnet_DLDL_200_0.2L1_20211204' \
    -d 'resnet34 DLDL 输入64核 标准方法 200轮学习率不减 优化了训练方式 使用了num_workers = 2 单输出模型 KL+0.2*L1'
