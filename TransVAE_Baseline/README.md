# Transformer CVAE Baseline
(unfinished projects, proper cleaning needed)

## For training:
```
python -m src.train.train_cvae
```

## For inferencing:
```
python INF.py
```

## 72 or 78
Be sure to precise wether you want to train/inference on 72 or 78 via INF/TRAIN_MODE var in INF.py, train_cave.py and training.py. (and also use the correct model in INF.py)

- 72 bits: 24 joints; 
- 78 bits: 24 joints + global rotation + transition. 

(For more details, refer to the report)