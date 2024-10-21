# Crop Yield Analysis and Prediction

## Project Overview
This project uses machine learning to analyze and predict crop yields, recommend suitable crops based on soil and weather data, recommend fertilizers, and predict crop prices. The goal is to help farmers optimize agricultural decisions and maximize productivity.

### Sub-Projects:
1. **Crop Recommendation Using Soil and Climatic Changes**
   - **Dataset**: [Crop Recommendation Dataset](https://data.mendeley.com/datasets/8v757rr4st/1)
   - **Models**: Logistic Regression, Random Forest, SVM, K-Nearest Neighbors (KNN)
   
2. **Crop Yield Prediction**
   - **Dataset**: [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/bassammakorvi/crop-production-based-on-different-states-of-india)
   - **Models**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting

3. **Fertilizer Recommendation**
   - **Dataset**: [Fertilizers Recommendation Dataset](https://www.kaggle.com/code/pazindushane/fertilizers-recommendation/input)
   - **Models**: Naive Bayes, Random Forest, Decision Tree, Support Vector Machine (SVM)

4. **Price Prediction of Agricultural Commodities**
   - **Dataset**: [Price of Agricultural Commodities Dataset](https://www.kaggle.com/datasets/anshtanwar/current-daily-price-of-various-commodities-india/data)
   - **Models**: Linear Regression, Ridge Regression, Lasso Regression, XGBoost

## Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Regression**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-url.git
    cd crop-yield-prediction
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Preprocess the datasets.
2. Train models for each sub-project and evaluate their performance.
3. Use the best model to make predictions.

## Assumptions
- The datasets are accurate and complete.
- Climate and soil conditions are consistent across the study period.
- The price dataset uses the unit "quintile," and crop yield uses "acres" and "tons."

## License
This project is licensed under the MIT License.

## Team Members

- [Thota Rahul](https://github.com/ThotaRahul) (AM.EN.U4CSE22257)
- [Kowshik Padala](https://github.com/KowshikPadala) (AM.EN.U4CSE22245)
- [Teja Sai Satwik](https://github.com/TejaSaiSatwik) (AM.EN.U4CSE22271)
- [Chetan Kalyan C](https://github.com/ChetanKalyanC) (AM.EN.U4CSE22216)

