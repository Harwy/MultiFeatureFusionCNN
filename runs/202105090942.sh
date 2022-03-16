echo "2021-05-09 09:42"
python E:/FeatureAgeNet/train.py -lr 0.0005 \
    -ep 200 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --outpath 'resnet34_coral_200' \
    -d 'resnet34 coral 标准方法 200轮学习率不减'
