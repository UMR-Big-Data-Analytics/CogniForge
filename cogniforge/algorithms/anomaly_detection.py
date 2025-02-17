from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans, MiniBatchKMeans
from autotsad.system.main import main as run_autotsad
from autotsad.system.main import register_autotsad_arguments
import argparse
import os,shutil

class AnomalyDetector(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def detect(self, data: np.ndarray) -> np.ndarray:
        pass

    def __reduce__(self):
        return {"name": self.name()}


class KMeansAnomalyDetector(AnomalyDetector):
    def __init__(self, window_size: int = 100, n_clusters: int = 50) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.window_size = window_size

    def name(self): 
        return "KMeans"

    def parameters(self):
        self.window_size = st.slider("Window size", 2, 1000, self.window_size)
        self.n_clusters = st.slider("Number of clusters", 2, 100, self.n_clusters)

    # def detect(self, data: np.ndarray) -> np.ndarray:
    #     windows = np.lib.stride_tricks.sliding_window_view(
    #         data, self.window_size, axis=0
    #     )
    #     windows = windows.reshape(-1, self.window_size * data.shape[1])
    #     km = KMeans(n_clusters=self.n_clusters,n_init='auto')
    #     labels = km.fit_predict(windows)
    #     anomaly_score = np.linalg.norm(windows - km.cluster_centers_[labels], axis=1)
    #     anomaly_score = (anomaly_score - anomaly_score.min()) / (
    #         anomaly_score.max() - anomaly_score.min()
    #     )
    #     return anomaly_score
    def detect(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)  # Reduce memory usage
        
        if data.shape[0] > 1_000_000:  
            self.window_size = min(self.window_size, 20)  

        windows = np.lib.stride_tricks.sliding_window_view(data, self.window_size, axis=0)
        windows = windows.reshape(-1, self.window_size * data.shape[1])

        batch_size = min(100_000, windows.shape[0])  # Process in batches

        km = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1000, random_state=42)
        
        # Train on all data in batches, but update the centroids globally
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start:start + batch_size]
            km.partial_fit(batch)  # Updates centroids incrementally

        # Once centroids are set, predict labels for the whole dataset
        labels = km.predict(windows)

        # Compute anomaly scores using the **global** centroids
        anomaly_score = np.linalg.norm(windows - km.cluster_centers_[labels], axis=1)

        anomaly_score = (anomaly_score - anomaly_score.min()) / (
            anomaly_score.max() - anomaly_score.min()
        )
        
        return anomaly_score


    def __reduce__(self):
        return {
            **super().__reduce__(),
            "window_size": self.window_size,
            "n_clusters": self.n_clusters,
        }

class AutoTSADAnomalyDetector(AnomalyDetector):
    def __init__(self):
        super().__init__()
        self._name = "AutoTSAD"
        self._use_gt_for_cleaning = False

    def name(self) -> str:
        return self._name

    def parameters(self):
        #check with team, what config can be updated from front end
        pass

    def detect(self, csv_path: str) -> np.ndarray:
        config_path = Path("autotsad.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        # Call the autotsad function
        parser = argparse.ArgumentParser()
        register_autotsad_arguments(parser)
        args = parser.parse_args(["--config-path", str(config_path), str(csv_path)])

        # Execute the autotsad run command
        anomaly_score = run_autotsad(args)
        if(os.path.isfile(csv_path)):
            os.remove(csv_path)
        return anomaly_score
    
    
class SpuckerCounter:
    def __init__(
        self,
        threshold: float = 70.0,
        ignore_begin_seconds: float = 1.0,
        ignore_end_seconds: float = 1.0,
        context_window: tuple[int, int] = (1, 5),
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_begin_seconds = ignore_begin_seconds
        self.ignore_end_seconds = ignore_end_seconds
        self.context_window = context_window

    def count(self, data: pd.Series) -> int:
        outliers = data > self.threshold

        # ignore outliers before ignore_begin_seconds and after ignore_end_seconds
        ignore_begin = outliers.index < self.ignore_begin_seconds
        ignore_end = outliers.index > outliers.index[-1] - self.ignore_end_seconds
        outliers[ignore_begin] = False
        outliers[ignore_end] = False

        # merge outliers that are close to each other (within context_window)
        count = 0
        local_count = 0
        last_outlier: int | None = None
        for i, outlier in enumerate(outliers):
            if outlier:
                if last_outlier is None or i - last_outlier > self.context_window[1]:
                    if local_count == 0 or local_count >= self.context_window[0]:
                        count += 1
                    else:
                        count -= 1
                    local_count = 1
                else:
                    local_count += 1
                last_outlier = i
        if local_count > 0 and local_count < self.context_window[0]:
            count -= 1

        return count

from sklearn.ensemble import IsolationForest

class IsolationForestAnomalyDetector(AnomalyDetector):
    def __init__(self, contamination: float = 0.05) -> None:
        super().__init__()
        self.contamination = contamination

    def name(self):
        return "Isolation Forest"

    def parameters(self):
        self.contamination = st.slider(
            "Contamination (Anomaly Ratio)", 0.01, 0.5, self.contamination, step=0.01
        )

    def detect(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)  # Reduce memory usage

        n_samples = min(100_000, data.shape[0])  # Use a subset for training
        sample_indices = np.random.choice(data.shape[0], n_samples, replace=False)
        training_data = data[sample_indices]

        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        iso_forest.fit(training_data)  # Train on a subset

        batch_size = 100_000  # Process in batches
        scores = np.zeros(data.shape[0], dtype=np.float32)  # Pre-allocate scores

        for start in range(0, data.shape[0], batch_size):
            batch = data[start:start + batch_size]
            scores[start:start + batch_size] = -iso_forest.decision_function(batch)

        # Normalize scores between 0 and 1
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = pd.read_csv(
        "data/2024-03-05_0002_10000011.csv",
        index_col=0,
        decimal=",",
        sep=";",
        encoding="latin",
    )

    df.plot()
    plt.show()

    counter = SpuckerCounter()
    print(counter.count(df.iloc[:, 0]))
