python -m latent_rationale.beer.dare \
    --model latent \
    --aspect 2 \
    --epochs 50 \
    --lr 0.0004 \
    --upper_bound 0.001 \
    --batch_size 256 \
    --train_path ./beer/reviews.aspect2.train.txt.gz \
    --dev_path ./beer/reviews.aspect2.heldout.txt.gz \
    --test_path ./beer/annotations.json \
    --scheduler exponential \
    --save_path ./dare_a2 \
    --dependent-z \
    --selection 0.07 --lasso 0.02
