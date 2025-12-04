import swanlab
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from typing import Any, Dict, Optional

class SwanLabLogger(Logger):
    def __init__(
        self,
        project: str,
        experiment_name: Optional[str] = None,
        workspace: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self._project = project
        self._experiment_name = experiment_name
        self._workspace = workspace
        self._config = config
        self._kwargs = kwargs
        self._run = None
        
        # Initialize immediately to ensure it's ready
        self._init_experiment()

    def _init_experiment(self):
        if self._run is None and rank_zero_only.rank == 0:
            self._run = swanlab.init(
                project=self._project,
                experiment_name=self._experiment_name,
                workspace=self._workspace,
                config=self._config,
                **self._kwargs
            )

    @property
    def name(self):
        return "SwanLabLogger"

    @property
    def version(self):
        return "0.1"

    @property
    def experiment(self):
        return self._run

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]):
        # SwanLab init handles config, but if we receive more here:
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._run:
            # Filter out non-scalar values just in case, though Lightning usually sends scalars
            # SwanLab log takes a dict.
            # If step is provided, we can pass it? 
            # Checking user example: swanlab.log({...})
            # Assuming swanlab.log(data, step=step) is valid or it tracks step internally.
            # If step is not supported by swanlab.log, we might need to check.
            # But most loggers support it.
            try:
                swanlab.log(metrics, step=step)
            except TypeError:
                # Fallback if step is not supported
                swanlab.log(metrics)

    @rank_zero_only
    def finalize(self, status: str):
        # Optional: swanlab.finish() if available
        pass
        
    def save(self):
        pass
