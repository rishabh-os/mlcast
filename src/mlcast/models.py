"""Abstract base classes and implementations for nowcasting models.

This module provides the core API for machine learning-based precipitation
nowcasting models, including abstract base classes and concrete implementations.
"""

import abc
from typing import Any, Optional
import xarray as xr
import numpy as np
import lightning as L
import pickle


class NowcastingModelBase(L.LightningModule, abc.ABC):
    """Abstract base class for precipitation nowcasting models.
    
    This class defines the standard interface that all nowcasting models
    must implement. It provides a consistent API for training, prediction,
    and model persistence across different ML approaches.
    
    Attributes:
        timestep_length: Time resolution of the model's predictions
    """
    
    timestep_length: Optional[np.timedelta64] = None
    
    def __init__(self):
        """Initialize the nowcasting model."""
        super().__init__()

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the trained model to disk.
        
        Args:
            path: File path where the model should be saved
            **kwargs: Additional arguments for model saving
            
        Note:
            This method needs to be implemented in concrete subclasses
            to handle model serialization.
        """
        # TODO: Implement model saving logic
        pass

    def load(self, path: str, **kwargs: Any) -> None:
        """Load a pre-trained model from disk.
        
        Args:
            path: File path to the saved model
            **kwargs: Additional arguments for model loading
            
        Note:
            This method needs to be implemented in concrete subclasses
            to handle model deserialization.
        """
        # TODO: Implement model loading logic
        pass

    @abc.abstractmethod
    def fit(self, da_rr: xr.DataArray, **kwargs: Any) -> None:
        """Train the nowcasting model on precipitation data.
        
        Args:
            da_rr: xarray DataArray containing precipitation radar data
                with time, latitude, and longitude dimensions
            **kwargs: Additional arguments for training (e.g., batch_size, epochs)
                
        Note:
            Concrete implementations should:
            1. Process the input data (scaling, temporal windowing)
            2. Train the underlying ML model
            3. Store scaling parameters and timestep information
        """
        pass

    @abc.abstractmethod
    def predict(self, da_rr: xr.DataArray, duration: str, **kwargs: Any) -> xr.DataArray:
        """Generate precipitation forecasts.
        
        Args:
            da_rr: xarray DataArray containing initial precipitation conditions
            duration: ISO 8601 duration string (e.g., "PT1H" for 1 hour)
                specifying how far into the future to predict
            **kwargs: Additional arguments for prediction (e.g., batch_size, device)
                
        Returns:
            xarray DataArray containing precipitation predictions with
            original spatial dimensions plus an "elapsed_time" dimension
            
        Note:
            Concrete implementations should:
            1. Process input data using stored scaling parameters
            2. Generate predictions using the trained model
            3. Return results in the original coordinate system
        """
        pass


class PersistenceNowcastingModel(NowcastingModelBase):
    """Persistence nowcasting model that repeats the last input timestep.
    
    This model implements a simple persistence forecast by repeating the last
    available timestep multiple times to create a forecast. It's commonly used
    as a baseline for comparing more sophisticated nowcasting models.
    """
    
    def __init__(self):
        """Initialize the persistence nowcasting model."""
        super().__init__()

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the persistence model to disk.
        
        Args:
            path: File path where the model should be saved
            **kwargs: Additional arguments (unused)
        """
        model_state = {
            'timestep_length': self.timestep_length
        }
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

    def load(self, path: str, **kwargs: Any) -> None:
        """Load a pre-trained persistence model from disk.
        
        Args:
            path: File path to the saved model
            **kwargs: Additional arguments (unused)
        """
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        self.timestep_length = model_state['timestep_length']

    def fit(self, da_rr: xr.DataArray, **kwargs: Any) -> None:
        """Train the persistence model (store timestep information).
        
        Args:
            da_rr: Precipitation radar data for training
            **kwargs: Additional training arguments (unused)
        """
        # Store timestep information
        if "time" in da_rr.dims and len(da_rr.time) > 1:
            self.timestep_length = da_rr.time.diff("time")[0]
        else:
            self.timestep_length = np.timedelta64(5, 'm')  # Default 5 minutes

    def predict(self, da_rr: xr.DataArray, duration: str, **kwargs: Any) -> xr.DataArray:
        """Generate persistence forecasts by repeating the last timestep.
        
        Args:
            da_rr: Initial precipitation conditions
            duration: Prediction duration (ISO 8601 format)
            **kwargs: Additional prediction arguments (unused)
            
        Returns:
            Precipitation predictions with elapsed_time dimension
            
        Raises:
            ValueError: If the model hasn't been trained yet or input is empty
        """
        if self.timestep_length is None:
            raise ValueError("Model must be trained before making predictions")
        
        if da_rr.size == 0:
            raise ValueError("Input data array is empty")
        
        # Simple duration parsing
        if duration == "PT1H":
            duration_td = np.timedelta64(1, 'h')
        elif duration == "PT30M":
            duration_td = np.timedelta64(30, 'm')
        elif duration == "PT15M":
            duration_td = np.timedelta64(15, 'm')
        else:
            raise ValueError(f"Unsupported duration format: {duration}")
            
        n_timesteps = int(duration_td / self.timestep_length)
        
        # Get the last timestep from input data
        if "time" in da_rr.dims:
            last_timestep = da_rr.isel(time=-1)
        else:
            last_timestep = da_rr
        
        # Repeat the last timestep n_timesteps times
        repeated_data = np.stack([last_timestep.values] * n_timesteps, axis=-1)
        
        # Construct output coordinates
        dims = list(last_timestep.dims) + ["elapsed_time"]
        coords = {dim: last_timestep[dim] for dim in last_timestep.dims}
        timestep_seconds = int(self.timestep_length.values / np.timedelta64(1, 's'))
        coords["elapsed_time"] = [np.timedelta64(timestep_seconds * i, 's') for i in range(n_timesteps)]
        
        return xr.DataArray(
            repeated_data,
            coords=coords,
            dims=dims
        )
