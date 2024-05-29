import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
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


target_variable = ['damage_grade', 'damage_overall_adjacent_building_risk', 'damage_overall_collapse',
                     'damage_overall_leaning', 'damage_roof', 'damage_corner_separation', 'damage_diagonal_cracking',
                     'damage_out_of_plane_failure', 'damage_gable_failure', 'damage_staircase',
                     'damage_out_of_plane_failure_walls_ncfr', 'damage_in_plane_failure', 'damage_foundation',

                     'has_damage_foundation', 'has_damage_roof', 'has_damage_corner_separation',
                     'has_damage_diagonal_cracking', 'has_damage_in_plane_failure', 'has_damage_out_of_plane_failure',
                     'has_damage_out_of_plane_walls_ncfr_failure', 'has_damage_gable_failure', 'has_damage_staircase']

for targets in target_variable:
    y = data[targets]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    dengelenmis_veri = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                                  pd.DataFrame(y_resampled, columns=[targets])], axis=1)

    dosya_adi = f"balanced_data_set{targets}.csv"
    dengelenmis_veri.to_csv(dosya_adi, index=False)


    data = pd.read_csv(dosya_adi, delimiter=',')


    target_column = targets


    features = data.drop(target_column, axis=1)
    target = data[target_column]


    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


    model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }


    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)


    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time


    best_model = grid_search.best_estimator_


    y_pred = best_model.predict(X_test)


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


    results_df.to_csv(f'prediction_results{target_column}.csv', index=False)
