from typing import Dict, List

import json
import os

from ray.tune.logger import LoggerCallback


class CustomLoggerCallback(LoggerCallback):
    """Custom logger interface"""

    def __init__(self, filename: str = "d:/log.txt"):
        self._trial_files = {}
        self._filename = filename

    def log_trial_start(self, trial: "Trial"):
        #trial_logfile = os.path.join(trial.logdir, self._filename)
        self._trial_files[trial] = open(self._filename, "at")

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        print(result)
        # if trial in self._trial_files:
        #     self._trial_files[trial].write(json.dumps(result))

    def on_trial_complete(self, iteration: int, trials: List["Trial"],
                          trial: "Trial", **info):
        if trial in self._trial_files:
            self._trial_files[trial].close()
            del self._trial_files[trial]