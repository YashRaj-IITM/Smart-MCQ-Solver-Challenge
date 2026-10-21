import numpy as np

def map_at_3(predictions, true_answers):
    scores = []
    for pred, true in zip(predictions, true_answers):
        pred_list = pred.split()
        if true in pred_list:
            rank = pred_list.index(true) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    return float(np.mean(scores))

def ensemble_predict(preds_list, weights):
    option_cols = ['A', 'B', 'C', 'D', 'E']
    n = len(preds_list[0])
    final_preds = []
    for i in range(n):
        scores = {c: 0.0 for c in option_cols}
        for preds, w in zip(preds_list, weights):
            ranked = preds[i].split()
            for rank, letter in enumerate(ranked):
                scores[letter] += w * (1.0 / (rank + 1))
        top3 = sorted(scores, key=scores.get, reverse=True)[:3]
        final_preds.append(" ".join(top3))
    return final_preds
