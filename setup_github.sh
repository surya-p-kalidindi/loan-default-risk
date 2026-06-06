#!/bin/bash
# Run from project root: bash setup_github.sh YOUR_GITHUB_USERNAME

USERNAME=$1
if [ -z "$USERNAME" ]; then
  echo "Usage: bash setup_github.sh YOUR_GITHUB_USERNAME"
  exit 1
fi

git init
git add .
git commit -m "Initial commit: Fannie Mae loan default risk model

- XGBoost mortality prediction: 90DPD AUC=0.791, Default AUC=0.765
- Time-based walk-forward validation (train 2010-2017, test 2019)
- SHAP feature importance + vintage analysis
- Platt-scaled calibrated probabilities
- Streamlit dashboard (5 tabs: risk drivers, geo, distributions, vintage, scorer)
- Claude API narrative generator (explainability, not decisioning)"

git branch -M main
git remote add origin https://github.com/$USERNAME/fannie-mae-default.git
git push -u origin main

echo "Done: https://github.com/$USERNAME/fannie-mae-default"
