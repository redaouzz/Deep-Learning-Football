# ⚽ Football Match Outcome Prediction with Deep Learning

Can we approach the predictive performance of **bookmakers’ strategies** using a neural network?  
This project explores that question by building and improving deep learning models to predict the outcome of football matches (Home Win / Draw / Away Win).

---

## 📖 Project Overview
Predicting football results is notoriously difficult due to high randomness and the many factors influencing outcomes (form, home advantage, injuries, etc.).  
Our objective was to **develop a deep learning model** capable of predicting match results, and compare its performance with professional benchmarks.

The project was carried out in 4 main stages:

1. **First Neural Network**
   - Built from a Kaggle football dataset.  
   - Preprocessing: merging multiple tables, selecting betting-related columns, filtering EPL 2020–2021.  
   - First model: simple 3-layer neural network → ~53% accuracy on 1,500 matches.  
   - Benchmark: *Datawin AI* claims 78% accuracy on a similar dataset.

2. **New Database via Web Scraping**
   - Scraped EPL match data (teams, stats, shots, matchweeks, multi-season).  
   - Consolidated into a new dataset for training and evaluation.  

3. **Model Improvements**
   - **SMOTE** for class rebalancing to address draw underrepresentation.  
   - Tested multiple architectures:
     - *Deeper*: 3×128 + 1×64 + softmax  
     - *Wider*: 3×256 + softmax  
     - *Deep & Regularized*: 2×128 + 1×64 + 1×32 + dropout + softmax  
   - **Regularization**:
     - L2 weight decay  
     - Dropout (0.2–0.5) to reduce overfitting  
   - **Validation**:
     - K-Fold cross-validation and iterative variants to maximize robustness.

4. **Conclusions & Perspectives**
   - Results improved with rebalancing and regularization but still below bookmaker-level performance.  
   - Future improvements:
     - Train on much larger datasets (multi-league, multi-season).  
     - Integrate additional features: defensive stats, possession, key players.  
     - Extend predictions to scorers, assists, and timing of goals.

