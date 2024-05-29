import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


data = pd.read_csv('damage_assessment.csv', delimiter=';')


X = data.drop(
    ['damage_grade', 'damage_overall_adjacent_building_risk', 'damage_overall_collapse', 'damage_overall_leaning',
     'damage_roof', 'damage_corner_separation', 'damage_diagonal_cracking',
     'damage_out_of_plane_failure', 'damage_gable_failure', 'damage_staircase',
     'damage_out_of_plane_failure_walls_ncfr', 'damage_in_plane_failure', 'damage_foundation', 'has_damage_foundation',
     'has_damage_roof', 'has_damage_corner_separation', 'has_damage_diagonal_cracking', 'has_damage_in_plane_failure',
     'has_damage_out_of_plane_failure', 'has_damage_out_of_plane_walls_ncfr_failure', 'has_damage_gable_failure',
     'has_damage_staircase'], axis=1)


target_variable  = ['damage_grade', 'damage_overall_adjacent_building_risk', 'damage_overall_collapse',
                     'damage_overall_leaning', 'damage_roof', 'damage_corner_separation', 'damage_diagonal_cracking',
                     'damage_out_of_plane_failure', 'damage_gable_failure', 'damage_staircase',
                     'damage_out_of_plane_failure_walls_ncfr', 'damage_in_plane_failure', 'damage_foundation',
                     'has_damage_foundation', 'has_damage_roof', 'has_damage_corner_separation',
                     'has_damage_diagonal_cracking', 'has_damage_in_plane_failure', 'has_damage_out_of_plane_failure',
                     'has_damage_out_of_plane_walls_ncfr_failure', 'has_damage_gable_failure', 'has_damage_staircase' ]

for targets  in target_variable :
    y = data[targets]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                               pd.DataFrame(y_resampled, columns=[targets ])], axis=1)

    file_name = f"balanced_data{targets}.csv"
    balanced_data.to_csv(file_name, index=False)


    data = pd.read_csv(file_name, delimiter=',')


    target_column = targets


    features = data.drop(target_column, axis=1)
    target = data[target_column]


    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


    model = RandomForestClassifier(random_state=42)


    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f'{target_column} Accuracy: {accuracy:.2f}')
    print(f'{target_column} Training Time: {training_time:.2f} seconds')


    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'{target_column} Confusion Matrix:\n{conf_matrix}')


    class_report = classification_report(y_test, y_pred)
    print(f'{target_column} Classification Report:\n{class_report}')


    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'{target_column} F1 Score: {f1:.2f}')


    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'{target_column} Precision: {precision:.2f}')


    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'{target_column} Recall: {recall:.2f}')


    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })


    results_df.to_csv(f'prediction_results_{target_column}.csv', index=False)

