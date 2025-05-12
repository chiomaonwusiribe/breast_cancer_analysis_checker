# Wine and Breast Cancer Analysis Web App

This project implements machine learning models for classification (wine and breast cancer datasets), clustering (breast cancer dataset), and a static web-based breast cancer prediction system using a Random Forest Classifier. The prediction system displays precomputed predictions for the top 5 features, served as a static site on InfinityFree. A PDF report summarizes findings.

## Setup (Local)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd wine_breast_cancer_ml
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train and save the model with precomputed predictions:
   ```bash
   python train_random_forest_model.py
   ```

4. Generate the PDF report:
   ```bash
   latexmk -pdf report.tex
   mv report.pdf static/report.pdf
   ```

5. Freeze the Flask app to static files:
   ```bash
   python freeze_app.py
   ```

6. Test locally:
   ```bash
   python -m http.server 8000 --directory build
   ```
   Open `http://localhost:8000` to view the static site.

## Deployment on InfinityFree (No Credit Card)

### Prerequisites
- A GitHub repository with the project code.
- Precompiled `report.pdf` and model files (`models/*.joblib`).
- Static files generated in `build/` using `freeze_app.py`.

### Deployment Steps
1. **Generate Static Files**:
   - Run:
     ```bash
     python freeze_app.py
     ```
   - Verify the `build/` directory contains `index.html`, `static/`, and other files.

2. **Sign Up for InfinityFree**:
   - Visit `https://infinityfree.com`.
   - Register with an email (no credit card required).
   - Choose a free subdomain (e.g., `breast-cancer.rf.gd`) or use a custom domain.

3. **Upload Files**:
   - Log in to InfinityFree’s control panel.
   - Use the File Manager or FTP (details in control panel) to upload the `build/` contents to the `htdocs` directory.
   - Ensure `index.html`, `static/report.pdf`, and `.htaccess` are in `htdocs`.

4. **Configure .htaccess**:
   - Place `.htaccess` in `htdocs` to handle routing.
   - Verify file permissions (default should work).

5. **Access the App**:
   - Visit your subdomain (e.g., `https://breast-cancer.rf.gd`).
   - Verify the prediction table, PDF toggle/download, and results tables.

## Project Structure

- `logistic_regression_wine.py`: Logistic Regression on wine dataset.
- `svm_wine.py`: SVM on wine dataset.
- `decision_tree_wine.py`: Decision Tree on wine dataset.
- `breast_cancer_experiment.py`: Classification on breast cancer dataset.
- `kmeans_breast_cancer.py`: K-Means on breast cancer dataset.
- `agglomerative_breast_cancer.py`: Agglomerative Clustering on breast cancer dataset.
- `dbscan_breast_cancer.py`: DBSCAN on breast cancer dataset.
- `gmm_breast_cancer.py`: GMM on breast cancer dataset.
- `train_random_forest_model.py`: Train and save Random Forest model with precomputed predictions.
- `app.py`: Flask app for static rendering.
- `freeze_app.py`: Freeze Flask app to static files.
- `templates/index.html`: Static page with prediction table.
- `static/`
  - `script.js`: JavaScript for PDF toggle.
  - `report.pdf`: Precompiled PDF report.
- `models/`
  - `random_forest_model.joblib`: Saved model (compressed).
  - `scaler.joblib`: Saved scaler (compressed).
  - `top_feature_indices.joblib`: Feature indices (compressed).
  - `top_feature_names.joblib`: Feature names (compressed).
  - `precomputed_predictions.joblib`: Precomputed predictions (compressed).
- `report.tex`: LaTeX source for PDF report.
- `requirements.txt`: Dependencies.
- `.htaccess`: InfinityFree routing config.
- `.gitignore`: Ignore virtual environments.
- `README.md`: Documentation.

## Results

### Wine Dataset (Classification)
- Logistic Regression: ~0.9815
- SVM: ~0.7593
- Decision Tree (random_state=42): ~0.9630
- Decision Tree (random_state=None): ~0.9444 (varies)

### Breast Cancer Dataset (Classification)
- Logistic Regression: ~0.9474
- SVM: ~0.9123
- Decision Tree (random_state=42): ~0.9181
- Decision Tree (random_state=None): ~0.9064 (varies)
- Random Forest: ~0.9591

### Breast Cancer Dataset (Clustering)
- K-Means (random_state=42): ~0.3512
- K-Means (random_state=None): ~0.3512 (varies)
- Agglomerative Clustering: ~0.3468
- DBSCAN (eps=3.0, min_samples=5): ~-0.2000
- Gaussian Mixture Model: ~0.3495

## Prediction System
- **Model**: Random Forest Classifier (~0.9591 accuracy).
- **Functionality**: Static table of precomputed predictions for top 5 features (Malignant/Benign with confidence).
- **Features**: Uses feature importance to display 5 features, served as static HTML.
- **Report**: PDF (`static/report.pdf`) summarizes findings, toggleable/downloadable.

## Observations
- **Report**: Concise PDF summarizes classification and prediction results.
- **Classification**: Random Forest excels for breast cancer (~0.9591); Logistic Regression for wine (~0.9815).
- **Prediction**: Static table shows precomputed predictions, suitable for demo purposes.