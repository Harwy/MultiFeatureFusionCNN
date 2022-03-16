echo "2021-05-15 12:00"
python E:/FeatureAgeNet/train_lite.py -lr 0.0005 \
    -ep 200 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --numworkers 2 \
    --outpath 'resnet34_coral_lite_20211125' \
    -d 'resnet34 lite: [3,3,3,3] coral 标准方法 200steps 200step*0.1 '
