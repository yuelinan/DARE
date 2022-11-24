Official implementation of NeurIPS'2022 paper "DARE:Disentanglement-Augmented Rationale Extraction".

## Mutual Information Estimation
We provide toy simulations in ./Simulated Studies/club_nce.ipynb to show the estimation performance of CLUB_NCE and other MI estimators.

## Multi-aspect Sentiment Analysis (Beer Advocate)
To train DARE on a single aspect, e.g. aspect 0 (look):
```
python -m latent_rationale.beer.dare \
    --model latent \
    --aspect 0 \
    --epochs 50 \
    --lr 0.00012 \
    --upper_bound 0.01 \
    --batch_size 200 \
    --train_path ./beer/reviews.aspect0.train.txt.gz \
    --dev_path ./beer/reviews.aspect0.heldout.txt.gz \
    --test_path ./beer/annotations.json \
    --scheduler exponential \
    --save_path ./dare_a0 \
    --dependent-z \
    --selection 0.13 --lasso 0.02

```
## Acknowledgements
The backbone of our code is referenced from codes released by [HardKuma](https://github.com/bastings/interpretable_predictions), [CLUB](https://github.com/Linear95/CLUB) and [SMILE](https://github.com/ermongroup/smile-mi-estimator). 
Thank you for their sharing !

## Citation
```
@inproceedings{yuedare,
  title={DARE: Disentanglement-Augmented Rationale Extraction},
  author={Yue, Linan and Liu, Qi and Du, Yichao and An, Yanqing and Wang, Li and Chen, Enhong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
