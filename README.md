# ML Network Intrusion Detector

## Project description
This educational cybersecurity project trains a Scikit-learn Random Forest classifier to categorize labeled network-flow records as **Benign (0)** or **Attack (1)**. It demonstrates data preparation, model training, evaluation, and feature-importance analysis.

**Important:** This is an offline learning prototype, not a live network-monitoring product or a guarantee of protection.

## Dataset
- **Dataset:** CIC-IDS2017, published by the Canadian Institute for Cybersecurity.
- **Official dataset page:** https://www.unb.ca/cic/datasets/ids-2017.html
- Download the machine-learning CSV files (for example, `MachineLearningCSV.zip`) and extract the CSV files into a folder named `data` beside the notebook or Python script.
- Review the dataset page and cite the dataset's associated research paper in academic submissions.

## Technologies used
- Python
- Pandas and NumPy
- Scikit-learn (Random Forest, train/test split, evaluation metrics)
- Matplotlib and Seaborn
- Jupyter Notebook

## Files
- `ML_Network_Intrusion_Detector.ipynb` — notebook implementation and visual evaluation.
- `ML_Network_Intrusion_Detector.py` — script version of the core model.
- `requirements.txt` — Python dependencies.
- `ProjectReport.docx` — project report template/documentation.

## Setup and run
1. Install Python 3.10 or later.
2. Download and extract the CIC-IDS2017 machine-learning CSV files into `data/`.
3. Create/activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the notebook:

   ```bash
   jupyter notebook ML_Network_Intrusion_Detector.ipynb
   ```

   Or run the script:

   ```bash
   python ML_Network_Intrusion_Detector.py
   ```

The notebook/script expects CSV files with a `Label` column. It maps `BENIGN` to 0 and other labels to 1. It uses up to 150,000 randomly sampled rows to keep training manageable; change `MAX_ROWS` if your hardware permits.

## Evaluation
The project reports accuracy, attack precision, attack recall, attack F1-score, a classification report, and a confusion matrix. Actual results are produced when you run the code on the dataset. Do not claim a particular accuracy until you have executed the experiment.

## Limitations
- CIC-IDS2017 is a benchmark dataset collected in 2017; current traffic may differ.
- Dataset artifacts, class imbalance, and related flows may affect results.
- A random split may overestimate generalization. Time-based or source-separated evaluation and external validation are recommended follow-ups.
- This prototype does not capture packets, inspect live traffic, or block attacks.

## Dataset citation
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” International Conference on Information Systems Security and Privacy (ICISSP), 2018.

