# hparams Short Summary

## 2L_GRU_1024_normed_bahdanau

    --attention=normed_bahdanau \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_units=1024 --unit_type=gru \
    --dropout=0.2 --num_layers=2 \

```log
step 900 overflow, stop early*******************************
step 900 lr 1 step-time 0.94s wps 5.95K ppl 417990255296330399744.00 gN 935534256287412.75 bleu 0.00, Fri Jan 18 04:53:24 2019
```

## 2L LuongScaled

    --attention:scaled_luong \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_units=128 --unit_type=lstm \
    --dropout=0.2 --num_layers=2 \

```log
# Fri Jan 18 11:53:54 > 13:40:35
    step-time 0.26s wps 21.86K ppl 14.77 gN 5.49
    dev ppl 13.85, dev bleu 16.2
    test ppl 12.86, test bleu 17.6
```

## 2L GRU 128

    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_units=128 --unit_type=gru \
    --dropout=0.2 --num_layers=2 \

```log
# Fri Jan 18 05:08:00 > 05:52:38 :
    step-time 0.16s wps 35.42K ppl 31.97 gN 4.69
    dev ppl 31.77, dev bleu 6.1,
    test ppl 37.03, test bleu 5.2
```

## 2L GRU 512

    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_units=512 --unit_type=gru \
    --dropout=0.2 --num_layers=2 \

```log
# 06:43:59 > 08:00:21
    step-time 0.31s wps 18.24K ppl 24.74 gN 6.29
    dev ppl 25.51, dev bleu 8.9
    test ppl 29.28, test bleu 7.7
```
