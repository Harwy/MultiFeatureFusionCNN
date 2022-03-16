echo "2021-05-08 12:52"
echo "90轮，每30轮减学习率"
python E:/FeatureAgeNet/train.py -lr 0.0005 \
    -ep 90 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --outpath 'resnet34_coral' \
    -d 'resnet34 coral 标准方法'
