echo "2021-05-17 12:00"
python E:/FeatureAgeNet/train_lite1.py  \
    -lr 0.0005 \
    -ep 300 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --outpath 'D:/Logs/resnet34_coral_lite1' \
    -d 'resnet34 lite: [3,3,3,3] coral 标准方法 300steps MultiStepLR[30, 100, 200] 使用kl_loss '
