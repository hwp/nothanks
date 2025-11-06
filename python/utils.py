import numpy as np

TAKE = 0
PASS = 1
ACTIONS = ["take", "pass"]


def to_binary_vector(cards):
    return [1.0 if c in cards else -1.0 for c in range(3, 36)]


def feature_from_player_state(player):
    return [(player.chips - 11.0) / 5.0, *to_binary_vector(player.cards)]


def _delta_mavg(scores, weight):
    if weight:
        delta_sum = 0
        weight_sum = 0
        for i, w in enumerate(weight):
            if w > 0:
                n = i + 1
                if n >= scores.shape[0]:
                    scores_n = (
                        np.ones(scores.shape[0] - 1) * scores[-1]
                    )  # repeat match end score
                else:
                    scores_n = np.concatenate(
                        (scores[n:], np.ones(n - 1) * scores[-1])
                    )  # repeat match end score
                delta_sum += (
                    scores_n - scores[:-1]
                ) * w  # no need for the match end score
                weight_sum += w
        return delta_sum / weight_sum
    else:
        return 0.0


def result_reward(result):
    return {"win": 1.0, "draw": 0.0, "loss": -1.0}[result]


def result_and_score_reward(config, score_history, result):
    result_reward_ = result_reward(result)

    others_scores = np.array([s["others"] for s in score_history])
    self_scores = np.array([s["score"] for s in score_history])
    rel_oavg = self_scores - others_scores.mean(axis=1)
    rel_obest = self_scores - others_scores.min(axis=1)

    return (
        result_reward_ * config["result_weight"]
        - _delta_mavg(self_scores, config["self_delta_mavg_weight"])
        * config["self_delta_weight"]
        - _delta_mavg(rel_oavg, config["rel_oavg_delta_mavg_weight"])
        * config["rel_oavg_delta_weight"]
        - _delta_mavg(rel_obest, config["rel_obest_delta_mavg_weight"])
        * config["rel_obest_delta_weight"]
    )
