import json
import pandas as pd

class DiagnosticLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log_step(self, step, history, true_value, candidate_predictions, selected_weights):
        """记录每一步所有候选技能的预测值"""
        record = {
            'step': step,
            'true': true_value,
            'candidates': candidate_predictions,  # {skill_name: predicted_value}
            'weights': selected_weights
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    @staticmethod
    def generate_report(diag_log_file):
        df = pd.read_json(diag_log_file, lines=True)
        # 分析各技能的平均绝对误差
        for skill in df['candidates'].iloc[0].keys():
            errors = []
            for _, row in df.iterrows():
                if skill in row['candidates']:
                    errors.append(abs(row['candidates'][skill] - row['true']))
            if errors:
                print(f"{skill}: MAE={np.mean(errors):.2f}")