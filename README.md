## Earthquake Damage Prediction using Machine Learning

This project uses machine learning models to predict earthquake damage based on building characteristics. The primary models used include decision trees, random forests, and gradient boosting.

## Project Structure

- damage_assessment.csv: The dataset containing building characteristics and damage-related features.
- data_visualization1.ipynb: Jupyter Notebook for visualizing the data (part 1).
- data_visualization2.ipynb: Jupyter Notebook for visualizing the data (part 2).
- decision_tree.py: Script to train and evaluate a decision tree model.
- gradient.py: Script to train and evaluate a gradient boosting model.
- random_forest.py: Script to train and evaluate a random forest model.


## Requirements

- Python 3.7+
- Jupyter Notebook
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
    bash
    git clone https://github.com/gsinem/earthquake-damage-prediction.git
    cd earthquake-damage-prediction
    

2. Create and activate the virtual environment:
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    


    

## Usage

1. To visualize the data, open and run the Jupyter Notebooks:
    bash
    jupyter notebook data_visualization1.ipynb
    jupyter notebook data_visualization2.ipynb
    

2. To train and evaluate the models, run the respective Python scripts:
    bash
    python decision_tree.py
    python random_forest.py
    python gradient.py
    

## Dataset

The damage_assessment.csv file contains the following columns:
- Building characteristics (e.g., age, height, material, etc.)
- Damage-related features (e.g., overall collapse, overall leaning, foundation damage, damage grade)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

