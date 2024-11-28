import pandas as pd
import numpy as np

# toggle rules
USE_GLUCOSE_RULE = True
USE_HBA1C_RULE = True
USE_AGE_RULE = True

# weights for each rule, if rules are disabled then these will be normalized in init
GLUCOSE_WEIGHT = 0.5
HBA1C_WEIGHT = 0.35
AGE_WEIGHT = 0.15

class ForwardChainingSystem:
    def __init__(self):
        self.rules = []
        self.weights = []
        
        # apply rules individually if they are enabled
        # ChatGPT was used to generate rules that could be toggled individually (the lambda functions here)
        # values used for the rules were adjusted manually based on correlations
        if USE_GLUCOSE_RULE:
            self.rules.append(
                lambda x: (
                    1 if x['blood_glucose_level'] >= 200 else  # 100% diabetes if above 200
                    0.8 if x['blood_glucose_level'] >= 160 else
                    0.3 if x['blood_glucose_level'] >= 140 else
                    0.1 if x['blood_glucose_level'] >= 100 else
                    0  # normal range is sub 100
                )
            )
            self.weights.append(GLUCOSE_WEIGHT)
            
        if USE_HBA1C_RULE:
            self.rules.append(
                lambda x: (
                    1 if x['HbA1c_level'] >= 7.0 else
                    0.8 if x['HbA1c_level'] >= 6.5 else
                    0.3 if x['HbA1c_level'] >= 6.0 else
                    0
                )
            )
            self.weights.append(HBA1C_WEIGHT)
            
        if USE_AGE_RULE:
            self.rules.append(
                lambda x: (
                    0.6 if x['age'] >= 80 else
                    0.4 if x['age'] >= 65 else
                    0.2 if x['age'] >= 45 else
                    0.1 if x['age'] >= 25 else
                    0
                )
            )
            self.weights.append(AGE_WEIGHT)
        
        # make sure enabled weights add up to 1.0
        if len(self.weights) > 0:
            total = 0
            for weight in self.weights:
                total = total + weight
            
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] / total
    
    def evaluate(self, patient_data):
        if not self.rules:  # if no rules are enabled
            return 0
        
        # Calculate weighted probability score
        probabilities = [rule(patient_data) for rule in self.rules]
        weighted_prob = sum(p * w for p, w in zip(probabilities, self.weights))
        
        # More strict threshold when age rule is enabled
        threshold = 0.65 if USE_AGE_RULE else 0.6
        return 1 if weighted_prob >= threshold else 0

class BackwardChainingSystem:
    def evaluate(self, patient_data):
        # initially assume no diabetes bc this is backward chained system
        has_diabetes = False
        
        # get values
        glucose = 0
        hba1c = 0
        age = 0
        if USE_GLUCOSE_RULE:
            glucose = patient_data['blood_glucose_level']
        if USE_HBA1C_RULE:
            hba1c = patient_data['HbA1c_level']
        if USE_AGE_RULE:
            age = patient_data['age']
        
        if USE_GLUCOSE_RULE and USE_HBA1C_RULE and USE_AGE_RULE:
            # all rules enabled
            if (glucose >= 200 and hba1c >= 6.5) or (glucose >= 220) or (hba1c >= 7.0):
                has_diabetes = True
            elif glucose >= 160 and hba1c >= 6.0 and age >= 45:
                has_diabetes = True
            elif glucose >= 180 and age >= 65:
                has_diabetes = True
                
        elif USE_GLUCOSE_RULE and USE_HBA1C_RULE:
            # glucose and HbA1c
            if glucose >= 200 and hba1c >= 6.5:
                has_diabetes = True
            elif glucose >= 220 or hba1c >= 7.0:
                has_diabetes = True
            elif glucose >= 160 and hba1c >= 6.0:
                has_diabetes = True
                
        elif USE_GLUCOSE_RULE and USE_AGE_RULE:
            # glucose and age
            if glucose >= 200:
                has_diabetes = True
            elif glucose >= 180 and age >= 45:
                has_diabetes = True
                
        elif USE_HBA1C_RULE and USE_AGE_RULE:
            # HbA1c and age
            if hba1c >= 7.0:
                has_diabetes = True
            elif hba1c >= 6.5 and age >= 45:
                has_diabetes = True
                
        elif USE_GLUCOSE_RULE:
            # only glucose rule
            if glucose >= 200:
                has_diabetes = True
            elif glucose >= 180:
                has_diabetes = True
                
        elif USE_HBA1C_RULE:
            # only HbA1c rule
            if hba1c >= 7.0:
                has_diabetes = True
            elif hba1c >= 6.5:
                has_diabetes = True
                
        elif USE_AGE_RULE:
            # only age rule (a bit dumb)
            if age >= 80:
                has_diabetes = True
        
        return 1 if has_diabetes else 0
    
# calculates some performance metrics that are mentioned in the slides
# precision, recall, f1 score and confusion matrix.
def evaluate_predictions(actual, predicted, system_name):
    # convert to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # calculate accuracy percentage
    accuracy = np.mean(actual == predicted) * 100
    
    # calculate precision
    true_positives = np.sum((actual == 1) & (predicted == 1))
    false_positives = np.sum((actual == 0) & (predicted == 1))
    if (true_positives + false_positives) > 0:  # so we dont div by 0
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    
    # calculate false negatives
    false_negative_condition = ((actual == 1) & (predicted == 0))
    false_negatives = np.sum(false_negative_condition)
    
    # calculate recall
    denominator = true_positives + false_negatives
    if denominator > 0:
        recall = true_positives / denominator
    else:
        recall = 0
    
    # calculate f1
    if precision + recall > 0:  # so we dont div by 0
        f1 = 2 * precision * recall / precision + recall
    else:
        f1 = 0
    
    # ChatGPT was used to generate this block of code calculating log loss
    # in the end, log loss isn't very meaningful since i am outputting binary 0/1 predictions instead of probabilities
    epsilon = 1e-15
    predicted_probs = np.clip(predicted, epsilon, 1 - epsilon)
    log_loss = -np.mean(actual * np.log(predicted_probs) + 
                       (1 - actual) * np.log(1 - predicted_probs))
    
    print(f"\n{system_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Log Loss: {log_loss:.3f}")
    
    # print confusion matrix
    tn = np.sum((actual == 0) & (predicted == 0))
    tp = true_positives
    fn = false_negatives
    fp = false_positives
    
    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"True Positives: {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")

def main():
    # load data (source: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    # initialize systems
    forward_system = ForwardChainingSystem()
    backward_system = BackwardChainingSystem()
    
    forward_predictions = []
    backward_predictions = []
    actual_diabetes = df['diabetes'].values
    
    for _, patient in df.iterrows():
        forward_predictions.append(forward_system.evaluate(patient))
        backward_predictions.append(backward_system.evaluate(patient))
    
    # convert predictions to numpy arrays
    forward_predictions = np.array(forward_predictions)
    backward_predictions = np.array(backward_predictions)
    
    # judge effectiveness of both systems
    evaluate_predictions(actual_diabetes, forward_predictions, "Forward Chaining")
    evaluate_predictions(actual_diabetes, backward_predictions, "Backward Chaining")

if __name__ == "__main__":
    main()
