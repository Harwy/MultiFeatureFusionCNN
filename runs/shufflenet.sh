echo "2021-11-30"
echo "200轮"
python E:/FeatureAgeNet/train_shufflenet.py -lr 0.0005 \
    -ep 200 \
    --num_classes 101 \
    --batch_size 64 \
    --data_mode 'rank' \
    --img_size 120 \
    --scale 2 \
    --c_tag 0.5 \
    --numworkers 2 \
    --SE \
    --outpath 'shufflenet_200_scale_2.0_SE' \
    -d 'shufflenetV2 scale 2.0 c_tag 0.5 SE True residual False 200轮均匀学习'
