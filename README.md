
## 1. **Create GitHub Repository & Branches**

```bash
# Initialize repo
git init iris-week-2
cd iris-week-2
git checkout -b main
echo "# Iris Homework" > README.md
git add .
git commit -m "Initial commit on main"

# Push to GitHub
gh repo create iris-week-2 --public --source=. --remote=origin
git push -u origin main

# Create dev branch
git checkout -b dev
git push -u origin dev
```

---

## 2. **Project Structure**

```
iris-week-2/
│
├── data/
│   └── iris.csv
│
├── src/
│   ├── preprocess.py
│   ├── evaluate.py
│   └── train.py
│
├── tests/
│   ├── test_preprocess.py
│   └── test_evaluate.py
│
├── .github/
│   └── workflows/
│       └── sanity-test.yml
│
├── requirements.txt
└── README.md
```
test
---

## 3. **Sample Code (Preprocess & Evaluate)**

### `src/preprocess.py`

```python
import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    return df
```

### `src/evaluate.py`

```python
from sklearn.metrics import accuracy_score

def evaluate_model(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
```

---

## 4. **Unit Tests using Pytest**

### `tests/test_preprocess.py`

```python
from src import preprocess

def test_load_data_shape():
    df = preprocess.load_data()
    assert df.shape[1] == 5  # 4 features + target
```

### `tests/test_evaluate.py`

```python
from src import evaluate

def test_accuracy_score():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    assert evaluate.evaluate_model(y_true, y_pred) == 1.0
```

---

## 5. **Install and List Requirements**

### `requirements.txt`

```
pandas
scikit-learn
pytest
cml
```

---

## 6. **GitHub Action for Sanity Test**

### `.github/workflows/sanity-test.yml`

```yaml
name: Sanity Test

on:
  pull_request:
    branches: [main]

jobs:
  sanity-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run Pytest
      run: |
        pytest tests/ --junitxml=results.xml > result.log

    - name: CML Report
      env:
        REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        pip install cml
        cml comment result.log
```

---

## 7. **Workflow Summary**

1. Push changes to `dev`
2. Create PR to `main`
3. GitHub Actions runs sanity test
4. CML posts the `pytest` report as a PR comment

---

## 8. **Push to `dev` and Create PR**

```bash
# On dev branch
git add .
git commit -m "Add evaluation and preprocess unit tests"
git push origin dev

# Create PR via CLI or GitHub UI
gh pr create --base main --head dev --title "Add unit tests and pipeline" --body "Includes pytest and CML pipeline."
```

---

