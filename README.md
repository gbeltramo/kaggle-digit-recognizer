# Kaggle: Digit Recognizer competition

## Overview
This repo provides Python code to obtain predictions for the [digit
recognizer](https://www.kaggle.com/competitions/digit-recognizer/overview)
Kaggle competition.

##  Setup
- conda env with `mamba`
- `kaggle.json` for API in `~/.kaggle/` and `chmod 600 $HOME/.kaggle/kaggle.json`
- kaggle command line
  1. `kaggle config set -n competition -v digit-recognizer` set competitions value
  2. `kaggle competitions files` list competition files
  3. Download the data in `$HOME/data/kaggle/digit-recognizer`

```bash
kaggle competitions download --path $HOME/data/kaggle/digit-recognizer
cd $HOME/data/kaggle/digit-recognizer && unzip digit-recognizer.zip
```

