### Multi-source unsupervised domain-adaptation for automatic sleep staging
#### *by: Yangxin Zhu; Shikui Tu; Lin Zhang; Lei Xu

## Prepare datasets
We used three public datasets in this study:
- [SleepEDF-39](https://www.physionet.org/content/sleep-edf/1.0.0/)
- [UCDDB](https://physionet.org/content/ucddb/1.0.0/)
- [ISRUC](https://sleeptight.isr.uc.pt/)


## Training model 
You can update different hyperparameters in the model by updating `config.py` file.

To train the model, use this command:
```
python trainer.py --target 'ucddb' --gpu 1
```

## Test model 

To test the model, use this command:

```
python test.py --target 'isruc'
```