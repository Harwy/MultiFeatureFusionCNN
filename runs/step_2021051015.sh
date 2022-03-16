echo "2021-05-10 15:42"
python E:/FeatureAgeNet/train_steps.py -lr 0.0005 \
    -ep 200 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --outpath 'resnet34_coral_200_202105101542' \
    -d 'resnet34 coral 标准方法 200轮学习率不变'
