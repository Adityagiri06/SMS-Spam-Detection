# SMS Spam Collection Dataset

This directory contains the UCI SMS Spam Collection dataset used for training.

## About the Dataset

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
- **File**: SMSSpamCollection (tab-separated, no extension)
- **Encoding**: UTF-8
- **Records**: 5,574 SMS messages
- **Classes**: 2 (Spam and Ham)
- **Language**: English

## Download

The dataset is automatically downloaded on first training run:
```bash
python train_model.py
```

## Format

The SMSSpamCollection file contains:
```
label\tmessage
ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
```

- **Column 1**: Label (ham or spam)
- **Column 2**: Raw SMS message text

## Data Split

- Training set: 80% (4,136 messages)
- Test set: 20% (1,035 messages)

## Citation

If you use this dataset, please cite:
```
Almeida, T. A., & Yamakami, A. (2016).
SMS Spam Collection v. 1.
UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
```

## License

Public domain - freely available for research and educational purposes.

---

**Note**: The downloaded `.zip` file is listed in `.gitignore` and will not be committed to git.
