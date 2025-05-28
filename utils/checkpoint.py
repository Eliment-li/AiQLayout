from ray.tune import Callback

from typing import Dict, List, Optional


class CheckPointCallback(Callback):
    def __init__(self):
        self._trial_ids = set()

    def on_trial_start(
            self, iteration: int, trials: List["Trial"], trial: "Trial", **info
    ):
        self._trial_ids.add(trial.trial_id)

    def get_state(self) -> Optional[Dict]:
        return {"trial_ids": self._trial_ids.copy()}

    def set_state(self, state: Dict) -> Optional[Dict]:
        self._trial_ids = state["trial_ids"]

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        print(f"iteration{iteration} Checkpoint saved at: {checkpoint}")
        # with open("d:/checkpoint_path.txt", "a") as f:
        #     f.write(f"Checkpoint for trial {trial.trial_id}: {checkpoint}\n")