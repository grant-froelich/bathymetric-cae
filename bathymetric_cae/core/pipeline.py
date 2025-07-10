"""
Main Processing Pipeline Module

This module orchestrates the complete bathymetric data processing pipeline,
including data loading, model training, inference, and result generation.

Author: Bathymetric CAE Team
License: MIT
"""

import gc
import json
import logging
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from skimage.metrics import structural_similarity as ssim

from .processor import BathymetricProcessor
from .model import AdvancedCAE
from ..visualization.visualizer import Visualizer
from ..utils.memory_utils import memory_monitor, force_garbage_collection
from ..utils.logging_utils import ContextLogger, log_function_call
from ..utils.file_utils import get_valid_files, validate_paths


class BathymetricCAEPipeline:
    """
    Main processing pipeline with enhanced features.
    
    This class orchestrates the complete workflow for bathymetric data processing
    using Convolutional Autoencoders, including:
    - Data preprocessing and validation
    - Model training or loading
    - Batch processing of files
    - Result visualization and reporting
    - Performance monitoring and optimization
    
    Attributes:
        config: Configuration object
        processor: Bathymetric data processor
        model_builder: Advanced CAE model builder
        visualizer: Visualization utilities
        logger: Logger instance
    """
    
    def __init__(self, config):
        """
        Initialize the processing pipeline.
        
        Args:
            config: Configuration object containing all pipeline parameters
        """
        self.config = config
        self.processor = BathymetricProcessor(config)
        self.model_builder = AdvancedCAE(config)
        self.visualizer = Visualizer()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("Pipeline initialized successfully")
    
    @log_function_call
    def run(
        self, 
        input_folder: str, 
        output_folder: str, 
        model_path: str,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete processing pipeline.
        
        Args:
            input_folder: Path to input bathymetric files
            output_folder: Path for output processed files
            model_path: Path to save/load model
            force_retrain: Force model retraining even if model exists
            
        Returns:
            Dict[str, Any]: Pipeline execution results
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If pipeline execution fails
        """
        try:
            with ContextLogger("Complete pipeline execution", self.logger):
                # Setup and validation
                self._setup_environment()
                self._validate_paths(input_folder, output_folder)
                
                # Get file list
                file_list = self._get_valid_files(input_folder)
                if not file_list:
                    raise ValueError(f"No valid files found in {input_folder}")
                
                self.logger.info(f"Found {len(file_list)} files to process")
                
                # Determine model parameters from sample file
                model_params = self._analyze_sample_file(file_list[0])
                
                # Train or load model
                model = self._get_or_train_model(
                    model_path, file_list, model_params, force_retrain
                )
                
                # Process files
                processing_results = self._process_files(model, file_list, output_folder)
                
                # Generate final report
                report = self._generate_final_report(
                    processing_results, model_params, file_list
                )
                
                self.logger.info("Pipeline completed successfully!")
                return report
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise RuntimeError(f"Pipeline execution failed: {e}")
    
    def _setup_environment(self):
        """Setup processing environment and directories."""
        with memory_monitor("Environment setup", self.logger):
            # Create output directories
            Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            Path("plots").mkdir(exist_ok=True)
            
            # Configure GPU if available
            try:
                from ..utils.gpu_utils import setup_gpu_environment
                gpu_results = setup_gpu_environment(self.config)
                self.logger.info(f"GPU setup results: {gpu_results}")
            except ImportError:
                self.logger.debug("GPU utilities not available")
    
    def _validate_paths(self, input_folder: str, output_folder: str):
        """Validate input and output paths."""
        if not Path(input_folder).exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Validate write permissions
        test_file = Path(output_folder) / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise PermissionError(f"Cannot write to output folder {output_folder}: {e}")
    
    def _get_valid_files(self, input_folder: str) -> List[Path]:
        """Get list of valid files to process."""
        try:
            from ..utils.file_utils import get_valid_files
            return get_valid_files(input_folder, self.config.supported_formats)
        except ImportError:
            # Fallback implementation
            input_path = Path(input_folder)
            valid_files = []
            
            for ext in self.config.supported_formats:
                valid_files.extend(input_path.glob(f"*{ext}"))
            
            return sorted(valid_files)
    
    def _analyze_sample_file(self, sample_file: Path) -> Dict[str, Any]:
        """Analyze sample file to determine model parameters."""
        self.logger.info(f"Analyzing sample file: {sample_file.name}")
        
        try:
            input_data, original_shape, geo_metadata = self.processor.preprocess_bathymetric_grid(sample_file)
            
            model_params = {
                'grid_size': input_data.shape[0],
                'channels': input_data.shape[-1],
                'original_shape': original_shape,
                'has_uncertainty': input_data.shape[-1] > 1,
                'sample_metadata': geo_metadata
            }
            
            self.logger.info(
                f"Model parameters determined: grid_size={model_params['grid_size']}, "
                f"channels={model_params['channels']}, uncertainty={model_params['has_uncertainty']}"
            )
            
            return model_params
            
        except Exception as e:
            self.logger.error(f"Failed to analyze sample file {sample_file}: {e}")
            raise
    
    def _get_or_train_model(
        self, 
        model_path: str, 
        file_list: List[Path], 
        model_params: Dict[str, Any],
        force_retrain: bool = False
    ):
        """Get existing model or train new one."""
        model_path = Path(model_path)
        
        # Try to load existing model if not forcing retrain
        if not force_retrain and model_path.exists():
            try:
                model = self.model_builder.load_model(str(model_path))
                self.logger.info(f"Loaded existing model from {model_path}")
                return model
            except Exception as e:
                self.logger.warning(f"Could not load existing model: {e}")
                self.logger.info("Creating new model instead")
        
        # Train new model
        return self._train_model(str(model_path), file_list, model_params)
    
    def _train_model(
        self, 
        model_path: str, 
        file_list: List[Path], 
        model_params: Dict[str, Any]
    ):
        """Train new model."""
        self.logger.info("Starting model training...")
        
        with memory_monitor("Model training", self.logger):
            # Create model
            model = self.model_builder.create_model(model_params['channels'])
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(file_list)
            
            # Setup callbacks
            callbacks = self.model_builder.create_callbacks(model_path)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self.visualizer.plot_training_history(
                history, 
                save_path='plots/training_history.png',
                show_plot=False
            )
            
            # Clean up training data
            del X_train, y_train
            force_garbage_collection()
            
            self.logger.info("Model training completed")
            return model
    
    def _prepare_training_data(self, file_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from file list."""
        self.logger.info("Preparing training data...")
        
        all_inputs = []
        all_targets = []
        
        for i, file_path in enumerate(file_list):
            try:
                if i % 10 == 0:
                    self.logger.info(f"Loading training file {i+1}/{len(file_list)}")
                
                input_data, _, _ = self.processor.preprocess_bathymetric_grid(file_path)
                
                # Prepare input and target data
                noisy_data = np.expand_dims(input_data, axis=0)
                clean_data = noisy_data[..., :1]  # Use first channel as target
                
                all_inputs.append(noisy_data)
                all_targets.append(clean_data)
                
            except Exception as e:
                self.logger.warning(f"Skipping {file_path.name}: {e}")
                continue
        
        if not all_inputs:
            raise ValueError("No valid training data found")
        
        X_train = np.vstack(all_inputs).astype(np.float32)
        y_train = np.vstack(all_targets).astype(np.float32)
        
        self.logger.info(f"Training data prepared: {X_train.shape} -> {y_train.shape}")
        
        return X_train, y_train
    
    def _process_files(
        self, 
        model, 
        file_list: List[Path], 
        output_folder: str
    ) -> Dict[str, Any]:
        """Process files using trained model."""
        output_path = Path(output_folder)
        
        successful_files = []
        failed_files = []
        processing_stats = []
        
        self.logger.info(f"Processing {len(file_list)} files...")
        
        for i, file_path in enumerate(file_list, 1):
            try:
                self.logger.info(f"Processing file {i}/{len(file_list)}: {file_path.name}")
                
                # Process single file
                stats = self._process_single_file(model, file_path, output_path)
                processing_stats.append(stats)
                successful_files.append(file_path.name)
                
                # Cleanup memory periodically
                if i % 10 == 0:
                    force_garbage_collection()
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                failed_files.append({
                    'filename': file_path.name,
                    'error': str(e)
                })
                continue
        
        results = {
            'successful_files': successful_files,
            'failed_files': failed_files,
            'processing_stats': processing_stats,
            'success_rate': len(successful_files) / len(file_list) * 100 if file_list else 0
        }
        
        self.logger.info(
            f"Processing complete: {len(successful_files)} successful, "
            f"{len(failed_files)} failed ({results['success_rate']:.1f}% success rate)"
        )
        
        return results
    
    def _process_single_file(
        self, 
        model, 
        file_path: Path, 
        output_path: Path
    ) -> Dict[str, Any]:
        """Process a single file and return processing statistics."""
        with memory_monitor(f"Processing {file_path.name}", self.logger):
            # Load and preprocess data
            input_data, original_shape, geo_metadata = self.processor.preprocess_bathymetric_grid(file_path)
            
            # Prepare for inference
            input_batch = np.expand_dims(input_data, axis=0).astype(np.float32)
            
            # Run inference
            prediction = model.predict(input_batch, verbose=0)[0, :, :, 0]
            
            # Calculate metrics
            original_depth = input_data[..., 0]
            ssim_score = self._calculate_ssim_safe(original_depth, prediction)
            
            # Handle uncertainty if available
            uncertainty_data = input_data[..., 1] if input_data.shape[-1] > 1 else None
            uncertainty_metrics = self._calculate_uncertainty_metrics(uncertainty_data, prediction)
            
            # Save results
            output_file = output_path / f"cleaned_{file_path.stem}.tif"
            self._save_processed_data(prediction, output_file, original_shape, geo_metadata)
            
            # Generate visualization
            plot_filename = f"plots/comparison_{file_path.stem}.png"
            self.visualizer.plot_comparison(
                original_depth, prediction, uncertainty_data, 
                filename=plot_filename, show_plot=False
            )
            
            # Return processing statistics
            stats = {
                'filename': file_path.name,
                'ssim': ssim_score,
                'original_shape': original_shape,
                'output_file': str(output_file),
                'plot_file': plot_filename,
                'processing_time': datetime.datetime.now().isoformat(),
                **uncertainty_metrics
            }
            
            self.logger.debug(f"Processed {file_path.name} - SSIM: {ssim_score:.4f}")
            return stats
    
    def _calculate_ssim_safe(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            if original.shape != cleaned.shape:
                self.logger.warning("Shape mismatch in SSIM calculation")
                return 0.0
            
            # Ensure data is in valid range
            data_range = float(max(cleaned.max() - cleaned.min(), 1e-8))
            
            return ssim(
                original.astype(np.float64),
                cleaned.astype(np.float64),
                data_range=data_range,
                gaussian_weights=True,
                win_size=min(7, min(original.shape))
            )
        except Exception as e:
            self.logger.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    def _calculate_uncertainty_metrics(
        self, 
        uncertainty_data: Optional[np.ndarray], 
        cleaned_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics."""
        if uncertainty_data is None:
            return {}
        
        try:
            return {
                'mean_uncertainty': float(np.mean(uncertainty_data)),
                'std_uncertainty': float(np.std(uncertainty_data)),
                'max_uncertainty': float(np.max(uncertainty_data)),
                'uncertainty_reduction': float(
                    np.mean(uncertainty_data) - np.mean(np.abs(uncertainty_data - cleaned_data))
                ),
                'uncertainty_correlation': float(
                    np.corrcoef(uncertainty_data.flatten(), cleaned_data.flatten())[0, 1]
                )
            }
        except Exception as e:
            self.logger.error(f"Error calculating uncertainty metrics: {e}")
            return {}
    
    def _save_processed_data(
        self, 
        data: np.ndarray, 
        output_path: Path, 
        original_shape: Tuple[int, int], 
        geo_metadata: Dict[str, Any]
    ):
        """Save processed data with geospatial metadata."""
        try:
            from osgeo import gdal
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine output format (default to GeoTIFF)
            driver_name = 'GTiff'
            if not output_path.suffix:
                output_path = output_path.with_suffix('.tif')
            
            # Create output dataset
            driver = gdal.GetDriverByName(driver_name)
            if driver is None:
                raise RuntimeError(f"Driver {driver_name} not available")
            
            dataset = driver.Create(
                str(output_path),
                original_shape[1], original_shape[0], 1,
                gdal.GDT_Float32
            )
            
            if dataset is None:
                raise RuntimeError(f"Failed to create dataset: {output_path}")
            
            # Set geospatial information
            if 'geotransform' in geo_metadata and geo_metadata['geotransform']:
                dataset.SetGeoTransform(geo_metadata['geotransform'])
            if 'projection' in geo_metadata and geo_metadata['projection']:
                dataset.SetProjection(geo_metadata['projection'])
            
            # Write data
            band = dataset.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))
            band.SetNoDataValue(-9999)
            
            # Set metadata
            processing_metadata = {
                'PROCESSING_DATE': datetime.datetime.now().isoformat(),
                'PROCESSING_SOFTWARE': 'Bathymetric CAE Pipeline',
                'MODEL_TYPE': 'Advanced Convolutional Autoencoder',
                'GRID_SIZE': str(self.config.grid_size),
                'SOURCE_FILE': geo_metadata.get('file_path', 'Unknown')
            }
            dataset.SetMetadata(processing_metadata, 'PROCESSING')
            
            # Flush and close
            dataset.FlushCache()
            dataset = None
            
            self.logger.debug(f"Successfully saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            raise
    
    def _generate_final_report(
        self, 
        processing_results: Dict[str, Any], 
        model_params: Dict[str, Any],
        file_list: List[Path]
    ) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        self.logger.info("Generating final report...")
        
        # Calculate summary statistics
        processing_stats = processing_results['processing_stats']
        if processing_stats:
            ssim_scores = [s['ssim'] for s in processing_stats if 'ssim' in s]
            summary_stats = {
                'mean_ssim': float(np.mean(ssim_scores)) if ssim_scores else 0.0,
                'std_ssim': float(np.std(ssim_scores)) if ssim_scores else 0.0,
                'min_ssim': float(np.min(ssim_scores)) if ssim_scores else 0.0,
                'max_ssim': float(np.max(ssim_scores)) if ssim_scores else 0.0,
                'median_ssim': float(np.median(ssim_scores)) if ssim_scores else 0.0
            }
        else:
            summary_stats = {}
        
        # Create comprehensive report
        report = {
            'pipeline_info': {
                'execution_date': datetime.datetime.now().isoformat(),
                'total_files': len(file_list),
                'successful_files': len(processing_results['successful_files']),
                'failed_files': len(processing_results['failed_files']),
                'success_rate': processing_results['success_rate']
            },
            'model_parameters': model_params,
            'summary_statistics': summary_stats,
            'processing_results': processing_results,
            'configuration': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        }
        
        # Save detailed report
        report_path = Path("processing_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate processing summary plot
        if processing_stats:
            self.visualizer.plot_processing_summary(
                processing_stats,
                filename='plots/processing_summary.png',
                show_plot=False
            )
        
        # Log summary
        self._log_final_summary(report)
        
        self.logger.info(f"Final report saved to: {report_path}")
        return report
    
    def _log_final_summary(self, report: Dict[str, Any]):
        """Log final summary to console."""
        pipeline_info = report['pipeline_info']
        summary_stats = report['summary_statistics']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("BATHYMETRIC CAE PROCESSING SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Execution Date: {pipeline_info['execution_date']}")
        self.logger.info(f"Total Files: {pipeline_info['total_files']}")
        self.logger.info(f"Successful: {pipeline_info['successful_files']}")
        self.logger.info(f"Failed: {pipeline_info['failed_files']}")
        self.logger.info(f"Success Rate: {pipeline_info['success_rate']:.1f}%")
        
        if summary_stats:
            self.logger.info(f"\nQuality Metrics:")
            self.logger.info(f"Mean SSIM: {summary_stats['mean_ssim']:.4f}")
            self.logger.info(f"SSIM Std: {summary_stats['std_ssim']:.4f}")
            self.logger.info(f"SSIM Range: {summary_stats['min_ssim']:.4f} - {summary_stats['max_ssim']:.4f}")
        
        self.logger.info(f"{'='*60}")
    
    def process_single_file_interactive(
        self, 
        file_path: str, 
        model_path: str,
        output_path: Optional[str] = None,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single file interactively with immediate visualization.
        
        Args:
            file_path: Path to file to process
            model_path: Path to trained model
            output_path: Optional output path
            show_plots: Whether to show plots
            
        Returns:
            Dict[str, Any]: Processing results
        """
        self.logger.info(f"Processing single file: {file_path}")
        
        try:
            # Load model
            model = self.model_builder.load_model(model_path)
            
            # Process file
            file_path = Path(file_path)
            input_data, original_shape, geo_metadata = self.processor.preprocess_bathymetric_grid(file_path)
            
            # Run inference
            input_batch = np.expand_dims(input_data, axis=0).astype(np.float32)
            prediction = model.predict(input_batch, verbose=0)[0, :, :, 0]
            
            # Calculate metrics
            original_depth = input_data[..., 0]
            ssim_score = self._calculate_ssim_safe(original_depth, prediction)
            uncertainty_data = input_data[..., 1] if input_data.shape[-1] > 1 else None
            
            # Visualize results
            if show_plots:
                self.visualizer.plot_comparison(
                    original_depth, prediction, uncertainty_data,
                    filename=f"single_file_comparison_{file_path.stem}.png",
                    show_plot=True,
                    title=f"Processing Results: {file_path.name}"
                )
                
                self.visualizer.plot_difference_map(
                    original_depth, prediction,
                    filename=f"single_file_difference_{file_path.stem}.png",
                    show_plot=True
                )
            
            # Save output if requested
            if output_path:
                output_path = Path(output_path)
                self._save_processed_data(prediction, output_path, original_shape, geo_metadata)
            
            results = {
                'filename': file_path.name,
                'ssim': ssim_score,
                'original_shape': original_shape,
                'has_uncertainty': uncertainty_data is not None,
                'processing_successful': True
            }
            
            self.logger.info(f"Single file processing completed - SSIM: {ssim_score:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process single file: {e}")
            return {
                'filename': file_path,
                'error': str(e),
                'processing_successful': False
            }


def run_pipeline_from_config(config_path: str, **kwargs) -> Dict[str, Any]:
    """
    Run pipeline from configuration file.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to override config
        
    Returns:
        Dict[str, Any]: Pipeline results
    """
    from ..config.config import Config
    
    # Load configuration
    config = Config.load(config_path)
    
    # Override with any provided arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create and run pipeline
    pipeline = BathymetricCAEPipeline(config)
    
    return pipeline.run(
        input_folder=config.input_folder,
        output_folder=config.output_folder,
        model_path=config.model_path
    )


def validate_pipeline_requirements() -> Dict[str, bool]:
    """
    Validate that all pipeline requirements are met.
    
    Returns:
        Dict[str, bool]: Validation results
    """
    requirements = {}
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        requirements['tensorflow'] = True
        requirements['tensorflow_version'] = tf.__version__
    except ImportError:
        requirements['tensorflow'] = False
    
    # Check GDAL
    try:
        from osgeo import gdal
        requirements['gdal'] = True
        requirements['gdal_version'] = gdal.__version__
    except ImportError:
        requirements['gdal'] = False
    
    # Check other dependencies
    for module in ['numpy', 'matplotlib', 'seaborn', 'scikit-image']:
        try:
            __import__(module)
            requirements[module] = True
        except ImportError:
            requirements[module] = False
    
    requirements['all_requirements_met'] = all(
        requirements.get(dep, False) 
        for dep in ['tensorflow', 'gdal', 'numpy', 'matplotlib', 'seaborn', 'scikit-image']
    )
    
    return requirements
    
    
def validate_installation():
    """
    Validate bathymetric CAE installation and dependencies.
    
    Returns:
        dict: Validation results
    """
    return validate_pipeline_requirements()