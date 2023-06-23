python -m latent_rationale.beer.movie \
    --model latent2 \
    --gpu_id 0 \
    --epochs 50 \
    --lr 6e-05 \
    --min_lr 5e-5 \
    --batch_size 32 \
    --layer rcnn \
    --max_len 512 \
    --train_path train.json \
    --dev_path dev.json \
    --test_path test.json \
    --scheduler exponential \
    --dependent-z \
    --selection 0.3 --lasso 0.02