"""
Enhanced Bathymetric CAE Processing Pipeline v2.0
Fixed the 'tuple' object has no attribute 'value' error by properly handling seafloor_type variable scope.
"""

import numpy as np
import logging
import datetime
import json
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
from osgeo import gdal
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model

from config.config import Config
from models.ensemble import BathymetricEnsemble
from core.adaptive_processor import AdaptiveProcessor
from core.quality_metrics import BathymetricQualityMetrics
from review.expert_system import ExpertReviewSystem
from processing.data_processor import BathymetricProcessor
from utils.memory_utils import memory_monitor, optimize_gpu_memory, log_memory_usage
from utils.visualization import create_enhanced_visualization, plot_training_history


class EnhancedBathymetricCAEPipeline:
    """Enhanced processing pipeline with TensorFlow version compatibility."""
    
    def __init__(self, config: Config):
        self.config = config
        self.ensemble = BathymetricEnsemble(config)
        self.adaptive_processor = AdaptiveProcessor()
        self.quality_metrics = BathymetricQualityMetrics()
        self.expert_review = ExpertReviewSystem() if config.enable_expert_review else None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Check TensorFlow version for compatibility
        self.tf_version = tf.__version__
        self.logger.info(f"TensorFlow version: {self.tf_version}")
        
        # Initialize processing statistics
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_processing_time': 0,
            'channel_distribution': {},
            'seafloor_type_distribution': {}
        }
    
    def run(self, input_folder: str, output_folder: str, model_path: str):
        """Run the enhanced processing pipeline with compatibility fixes."""
        try:
            self.logger.info("="*60)
            self.logger.info("ENHANCED BATHYMETRIC CAE PIPELINE v2.0")
            self.logger.info("="*60)
            
            # Setup
            self._setup_environment()
            self._validate_paths(input_folder, output_folder)
            
            # Get file list
            file_list = self._get_valid_files(input_folder)
            if not file_list:
                raise ValueError(f"No valid files found in {input_folder}")
            
            self.logger.info(f"Found {len(file_list)} files to process")
            self._log_file_summary(file_list)
            
            # Analyze data consistency
            channel_analysis = self._analyze_data_channels(file_list)
            self.logger.info(f"Data channel analysis: {channel_analysis}")
            
            # Train or load ensemble with compatibility fixes
            ensemble_models = self._get_or_train_ensemble(model_path, file_list, channel_analysis)
            
            # Process files with enhanced features
            self._process_files_enhanced(ensemble_models, file_list, output_folder)
            
            # Generate comprehensive reports
            self._generate_comprehensive_reports()
            
            # Generate expert review report
            if self.expert_review:
                self._generate_expert_review_report()
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.debug("Full traceback:", exc_info=True)
            raise
    
    def _setup_environment(self):
        """Setup processing environment with optimization."""
        self.logger.info("Setting up processing environment...")
        
        # Fix HDF5 issues
        try:
            import os
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        except:
            pass
        
        # Optimize GPU memory
        optimize_gpu_memory()
        
        # Create output directories
        directories = [
            self.config.output_folder,
            "logs",
            "plots",
            "expert_reviews",
            "models",
            "temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Log environment info
        log_memory_usage("Environment setup")
        
        # Check TensorFlow setup
        gpu_count = len(tf.config.list_physical_devices('GPU'))
        self.logger.info(f"TensorFlow setup: {gpu_count} GPU(s) available")
        
        if gpu_count > 0:
            self.logger.info("GPU acceleration enabled")
        else:
            self.logger.info("Using CPU processing")
    
    def _validate_paths(self, input_folder: str, output_folder: str):
        """Validate input and output paths."""
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_folder}")
        
        # Create output directory
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input folder: {input_path.absolute()}")
        self.logger.info(f"Output folder: {output_path.absolute()}")
    
    def _get_valid_files(self, input_folder: str) -> List[Path]:
        """Get list of valid bathymetric files to process."""
        input_path = Path(input_folder)
        valid_files = []
        
        self.logger.info(f"Scanning for supported formats: {self.config.supported_formats}")
        
        for ext in self.config.supported_formats:
            pattern_files = list(input_path.glob(f"*{ext}"))
            valid_files.extend(pattern_files)
            if pattern_files:
                self.logger.debug(f"Found {len(pattern_files)} {ext} files")
        
        # Remove duplicates and sort
        valid_files = sorted(list(set(valid_files)))
        
        return valid_files
    
    def _log_file_summary(self, file_list: List[Path]):
        """Log summary of files to be processed."""
        file_sizes = {}
        total_size = 0
        
        for file_path in file_list:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                ext = file_path.suffix.lower()
                if ext not in file_sizes:
                    file_sizes[ext] = {'count': 0, 'size': 0}
                file_sizes[ext]['count'] += 1
                file_sizes[ext]['size'] += size_mb
                total_size += size_mb
        
        self.logger.info("File summary:")
        for ext, info in file_sizes.items():
            self.logger.info(f"  {ext}: {info['count']} files")
        self.logger.info(f"  Total size: {total_size:.1f} MB")
    
    def _analyze_data_channels(self, file_list: List[Path]) -> Dict:
        """Analyze data channel consistency across files."""
        self.logger.info("Analyzing data channel consistency...")
        
        channel_info = {}
        failed_files = []
        processor = BathymetricProcessor(self.config)
        
        for file_path in file_list:
            try:
                input_data, _, _ = processor.preprocess_bathymetric_grid(file_path)
                ext = file_path.suffix.lower()
                channels = input_data.shape[-1]
                shape = input_data.shape[:-1]
                
                if ext not in channel_info:
                    channel_info[ext] = {'channels': [], 'shapes': [], 'files': []}
                
                channel_info[ext]['channels'].append(channels)
                channel_info[ext]['shapes'].append(shape)
                channel_info[ext]['files'].append(file_path.name)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path.name}: {e}")
                failed_files.append(file_path.name)
        
        # Log channel analysis
        for ext, info in channel_info.items():
            unique_channels = list(set(info['channels']))
            self.logger.info(f"  {ext}: {unique_channels} channels")
        
        max_channels = max([max(info['channels']) for info in channel_info.values()]) if channel_info else 1
        
        return {
            'max_channels': max_channels,
            'channel_info': channel_info,
            'failed_files': failed_files
        }
    
    def _get_or_train_ensemble(self, model_path: str, file_list: List[Path], 
                              channel_analysis: Dict) -> List:
        """Train or load ensemble models with proper channel handling."""
        model_file = Path(model_path)
        max_channels = channel_analysis.get('max_channels', 1)
        
        # Check for existing ensemble models
        ensemble_files = []
        for i in range(self.config.ensemble_size):
            ensemble_path = model_file.with_name(f"{model_file.stem}_ensemble_{i}.h5")
            if ensemble_path.exists():
                ensemble_files.append(ensemble_path)
        
        if len(ensemble_files) == self.config.ensemble_size:
            self.logger.info(f"Loading existing ensemble from {len(ensemble_files)} files")
            try:
                models = []
                for ensemble_path in ensemble_files:
                    model = load_model(str(ensemble_path))
                    models.append(model)
                    self.logger.debug(f"Loaded model: {ensemble_path}")
                return models
            except Exception as e:
                self.logger.warning(f"Failed to load ensemble: {e}")
        
        # Train new ensemble
        self.logger.info("Training new ensemble")
        return self._train_ensemble(file_list, model_path, max_channels)
    
    def _train_ensemble(self, file_list: List[Path], model_path: str, channels: int) -> List:
        """Train ensemble models with enhanced features."""
        self.logger.info("Training new ensemble models...")
        self.logger.info(f"Training ensemble for {channels} channels")
        
        # Create ensemble
        ensemble_models = self.ensemble.create_ensemble(channels=channels)
        
        # Prepare training data
        self.logger.info(f"Preparing training data with {channels} channels...")
        training_data = self._prepare_training_data(file_list, channels)
        
        if len(training_data) == 0:
            raise ValueError("No valid training data found")
        
        self.logger.info(f"Training data prepared: {training_data[0].shape} -> {training_data[1].shape}")
        self.logger.info(f"Successfully processed {len(training_data[0])}/{len(file_list)} files for training")
        
        # Train each model in the ensemble
        model_path_obj = Path(model_path)
        trained_models = []
        
        for i, model in enumerate(ensemble_models):
            self.logger.info(f"Training ensemble model {i+1}/{len(ensemble_models)}")
            
            # Generate model save path
            ensemble_model_path = model_path_obj.with_name(f"{model_path_obj.stem}_ensemble_{i}.h5")
            self.logger.debug(f"Using model format: {ensemble_model_path}")
            
            # Setup callbacks with legacy format
            callbacks_list = [
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            ]
            
            # Use legacy ModelCheckpoint format for compatibility
            self.logger.info("Using legacy ModelCheckpoint format")
            checkpoint_callback = callbacks.ModelCheckpoint(
                str(ensemble_model_path),
                save_best_only=True,
                monitor='val_loss'
            )
            callbacks_list.append(checkpoint_callback)
            
            # Train model with memory monitoring
            with memory_monitor(f"Training ensemble model {i+1}"):
                history = model.fit(
                    training_data[0], training_data[1],
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    validation_split=self.config.validation_split,
                    callbacks=callbacks_list,
                    verbose=0
                )
            
            trained_models.append(model)
            
            # Save training history
            if history and hasattr(history, 'history'):
                plot_training_history(history, f"training_history_ensemble_{i}.png")
        
        training_time = self.processing_stats.get('total_processing_time', 0)
        self.logger.info(f"Ensemble training completed in {training_time:.1f} seconds")
        
        return trained_models
    
    def _prepare_training_data(self, file_list: List[Path], target_channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with consistent channel count."""
        processor = BathymetricProcessor(self.config)
        inputs = []
        targets = []
        
        for file_path in file_list:
            try:
                input_data, _, _ = processor.preprocess_bathymetric_grid(file_path)
                
                # Normalize channels
                input_data = self._normalize_channels(input_data, target_channels)
                
                # Get seafloor type for adaptive processing
                depth_data = input_data[..., 0]
                seafloor_type_enum = self.adaptive_processor.seafloor_classifier.classify(depth_data)
                adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
                
                self.logger.debug(f"Classified as {seafloor_type_enum.value} with confidence {adaptive_params.get('confidence', 1.0):.2f}")
                self.logger.info(f"Adaptive parameters for {seafloor_type_enum.value}: compression={adaptive_params.get('compression_ratio', 'N/A')}, grid_size={adaptive_params.get('grid_size', 'N/A')}")
                
                # For training, target is the first channel (depth)
                target_data = input_data[..., 0:1]
                
                inputs.append(input_data)
                targets.append(target_data)
                
                self.logger.debug(f"Loaded training data from {file_path.name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load training data from {file_path.name}: {e}")
                continue
        
        if not inputs:
            return np.array([]), np.array([])
        
        return np.array(inputs), np.array(targets)
    
    def _normalize_channels(self, input_data: np.ndarray, target_channels: int) -> np.ndarray:
        """Normalize input data to have target number of channels."""
        current_channels = input_data.shape[-1]
        
        if current_channels == target_channels:
            return input_data
        
        elif current_channels < target_channels:
            # Need to add channels
            if target_channels == 2 and current_channels == 1:
                # Add uncertainty channel (estimate as percentage of depth)
                depth_channel = input_data[..., 0]
                uncertainty_channel = np.abs(depth_channel * 0.05) + 0.1
                return np.stack([depth_channel, uncertainty_channel], axis=-1)
            else:
                # General padding with zeros
                padding_shape = input_data.shape[:-1] + (target_channels - current_channels,)
                padding = np.zeros(padding_shape, dtype=input_data.dtype)
                return np.concatenate([input_data, padding], axis=-1)
        
        else:
            # Need to reduce channels - take first target_channels
            return input_data[..., :target_channels]
    
    def _apply_adaptive_preprocessing(self, input_data: np.ndarray, 
                                    adaptive_params: Dict) -> np.ndarray:
        """Apply adaptive preprocessing based on seafloor type."""
        # Apply smoothing based on seafloor type
        smoothing_factor = adaptive_params.get('smoothing_factor', 0.5)
        
        if smoothing_factor > 0:
            # Apply Gaussian smoothing to each channel
            sigma = smoothing_factor * 2
            processed_data = input_data.copy()
            
            for channel in range(input_data.shape[-1]):
                processed_data[..., channel] = ndimage.gaussian_filter(
                    processed_data[..., channel], sigma=sigma
                )
            
            return processed_data
        
        return input_data
    
    def _process_files_enhanced(self, ensemble_models: List, file_list: List[Path], 
                               output_folder: str):
        """Process files using enhanced ensemble approach."""
        output_path = Path(output_folder)
        processor = BathymetricProcessor(self.config)
        
        successful_files = []
        failed_files = []
        processing_stats = []
        
        self.logger.info(f"Processing {len(file_list)} files with enhanced pipeline...")
        
        # Determine channel count from ensemble
        if ensemble_models:
            try:
                model_input_shape = ensemble_models[0].input_shape
                expected_channels = model_input_shape[-1]
                self.logger.info(f"Ensemble expects {expected_channels} channels")
            except:
                expected_channels = 1
                self.logger.warning("Could not determine expected channels, using 1")
        else:
            expected_channels = 1
        
        processing_start_time = datetime.datetime.now()
        
        for i, file_path in enumerate(file_list, 1):
            file_start_time = datetime.datetime.now()
            
            try:
                self.logger.info(f"Processing file {i}/{len(file_list)}: {file_path.name}")
                
                # Process single file with enhanced features
                stats = self._process_single_file_enhanced(
                    processor, file_path, output_path, expected_channels
                )
                processing_stats.append(stats)
                successful_files.append(file_path.name)
                
                # Update processing statistics
                self.processing_stats['files_processed'] += 1
                
                # Track seafloor types
                seafloor_type_value = stats.get('seafloor_type', 'unknown')
                self.processing_stats['seafloor_type_distribution'][seafloor_type_value] = \
                    self.processing_stats['seafloor_type_distribution'].get(seafloor_type_value, 0) + 1
                
                # Check if flagging for expert review is needed
                if (self.expert_review and 
                    stats.get('composite_quality', 1.0) < self.config.quality_threshold):
                    self._flag_for_expert_review(file_path.name, stats)
                
                file_end_time = datetime.datetime.now()
                file_duration = (file_end_time - file_start_time).total_seconds()
                self.logger.info(f"Completed {file_path.name} in {file_duration:.1f}s "
                               f"(Quality: {stats.get('composite_quality', 0):.3f})")
                
                # Memory management
                if i % 5 == 0:
                    gc.collect()
                    tf.keras.backend.clear_session()
                    log_memory_usage(f"After processing {i} files")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                failed_files.append(file_path.name)
                self.processing_stats['files_failed'] += 1
                
                # Cleanup on error
                gc.collect()
                continue
        
        processing_end_time = datetime.datetime.now()
        total_duration = (processing_end_time - processing_start_time).total_seconds()
        self.processing_stats['total_processing_time'] = total_duration
        
        # Final cleanup
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Generate enhanced summary report
        self._generate_enhanced_summary_report(successful_files, failed_files, processing_stats)
        
        self.logger.info(f"Processing completed: {len(successful_files)} successful, "
                        f"{len(failed_files)} failed in {total_duration:.1f}s")
    
    def _process_single_file_enhanced(self, processor, file_path: Path, output_path: Path,
                                     expected_channels: int) -> Dict:
        """Process a single file with enhanced ensemble approach and consistent channels.
        
        FIXED: Properly handle seafloor_type variable scope to prevent 'tuple' object has no attribute 'value' error.
        """
        
        with memory_monitor(f"Processing {file_path.name}"):
            # Load and preprocess data
            input_data, original_shape, geo_metadata = processor.preprocess_bathymetric_grid(file_path)
            
            # Normalize channels to match ensemble expectations
            input_data = self._normalize_channels(input_data, expected_channels)
            
            # Get adaptive processing parameters
            depth_data = input_data[..., 0]
            seafloor_type_enum = self.adaptive_processor.seafloor_classifier.classify(depth_data)
            seafloor_type_value = seafloor_type_enum.value  # Store string value immediately to prevent scope issues
            
            adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
            adaptive_params['seafloor_type'] = seafloor_type_value  # Use stored string value
            
            # Apply adaptive preprocessing
            enhanced_input = self._apply_adaptive_preprocessing(input_data, adaptive_params)
            
            # Prepare for ensemble inference
            input_batch = np.expand_dims(enhanced_input, axis=0).astype(np.float32)
            
            # Run ensemble prediction
            ensemble_prediction, ensemble_metrics = self.ensemble.predict_ensemble(
                input_batch, adaptive_params
            )
            
            # Extract final prediction
            final_prediction = ensemble_prediction[0, :, :, 0]
            
            # Calculate additional metrics
            original_depth = input_data[..., 0]
            uncertainty_data = input_data[..., 1] if input_data.shape[-1] > 1 else None
            
            # Enhanced quality assessment
            quality_metrics = self._calculate_comprehensive_quality_metrics(
                original_depth, final_prediction, uncertainty_data
            )
            
            # Combine all metrics
            all_metrics = {**ensemble_metrics, **quality_metrics}
            
            # Save results with enhanced metadata
            output_file = output_path / f"enhanced_{file_path.name}"
            self._save_enhanced_results(
                final_prediction, output_file, original_shape, 
                geo_metadata, all_metrics, adaptive_params
            )
            
            # Generate enhanced visualization
            try:
                create_enhanced_visualization(
                    original_depth, final_prediction, uncertainty_data, 
                    all_metrics, file_path, adaptive_params
                )
            except Exception as e:
                self.logger.warning(f"Failed to create visualization for {file_path.name}: {e}")
            
            # Return comprehensive statistics - FIXED: Use stored string value instead of enum.value
            stats = {
                'filename': file_path.name,
                'processing_time': datetime.datetime.now().isoformat(),
                'seafloor_type': seafloor_type_value,  # Use stored string value to prevent tuple error
                'adaptive_params': adaptive_params,
                'input_channels': input_data.shape[-1],
                'output_shape': final_prediction.shape,
                **all_metrics
            }
            
            return stats
    
    def _calculate_comprehensive_quality_metrics(self, original: np.ndarray, 
                                               cleaned: np.ndarray,
                                               uncertainty: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive quality metrics with error handling."""
        try:
            # Calculate core quality metrics
            metrics = {
                'ssim': self.quality_metrics.calculate_ssim(original, cleaned),
                'roughness': self.quality_metrics.calculate_roughness(cleaned),
                'feature_preservation': self.quality_metrics.calculate_feature_preservation(original, cleaned),
                'consistency': self.quality_metrics.calculate_depth_consistency(cleaned),
                'hydrographic_compliance': self.quality_metrics.calculate_hydrographic_standards_compliance(cleaned)
            }
            
            # Calculate uncertainty metrics if available
            if uncertainty is not None:
                uncertainty_metrics = self._calculate_uncertainty_metrics(uncertainty, cleaned)
                metrics.update(uncertainty_metrics)
            
            # Calculate composite quality score
            metrics['composite_quality'] = (
                self.config.ssim_weight * metrics['ssim'] +
                self.config.roughness_weight * max(0, 1.0 - metrics['roughness']) +
                self.config.feature_preservation_weight * metrics['feature_preservation'] +
                self.config.consistency_weight * metrics['consistency']
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            return {
                'composite_quality': 0.5,
                'ssim': 0.0,
                'roughness': 1.0,
                'feature_preservation': 0.0,
                'consistency': 0.0,
                'hydrographic_compliance': 0.0
            }
    
    def _calculate_uncertainty_metrics(self, uncertainty_data: np.ndarray, 
                                     cleaned_data: np.ndarray) -> Dict:
        """Calculate comprehensive uncertainty metrics with error handling."""
        try:
            return {
                'mean_uncertainty': float(np.mean(uncertainty_data)),
                'std_uncertainty': float(np.std(uncertainty_data)),
                'max_uncertainty': float(np.max(uncertainty_data)),
                'uncertainty_reduction': float(np.mean(uncertainty_data) - 
                                             np.mean(np.abs(uncertainty_data - cleaned_data))),
                'uncertainty_correlation': float(np.corrcoef(uncertainty_data.flatten(), 
                                                           cleaned_data.flatten())[0, 1])
            }
        except Exception as e:
            self.logger.error(f"Error calculating uncertainty metrics: {e}")
            return {}
    
    def _save_enhanced_results(self, data: np.ndarray, output_path: Path, 
                             original_shape: Tuple[int, int], geo_metadata: Dict,
                             quality_metrics: Dict, adaptive_params: Dict):
        """Save enhanced results with comprehensive metadata and error handling."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine output format
            ext = output_path.suffix.lower()
            if ext == '.bag':
                driver_name = 'BAG'
            elif ext in ['.tif', '.tiff']:
                driver_name = 'GTiff'
            elif ext == '.asc':
                driver_name = 'AAIGrid'
            else:
                output_path = output_path.with_suffix('.tif')
                driver_name = 'GTiff'
            
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
            
            # Set enhanced metadata
            processing_metadata = {
                'PROCESSING_DATE': datetime.datetime.now().isoformat(),
                'PROCESSING_SOFTWARE': 'Enhanced Bathymetric CAE v2.0',
                'MODEL_TYPE': 'Ensemble Convolutional Autoencoder',
                'TENSORFLOW_VERSION': self.tf_version,
                'ENSEMBLE_SIZE': str(self.config.ensemble_size),
                'GRID_SIZE': str(self.config.grid_size),
                'COMPOSITE_QUALITY': f"{quality_metrics.get('composite_quality', 0):.4f}",
                'SSIM_SCORE': f"{quality_metrics.get('ssim', 0):.4f}",
                'FEATURE_PRESERVATION': f"{quality_metrics.get('feature_preservation', 0):.4f}",
                'HYDROGRAPHIC_COMPLIANCE': f"{quality_metrics.get('hydrographic_compliance', 0):.4f}",
                'SEAFLOOR_TYPE': adaptive_params.get('seafloor_type', 'unknown'),
                'ADAPTIVE_PROCESSING': 'TRUE' if self.config.enable_adaptive_processing else 'FALSE',
                'CONSTITUTIONAL_CONSTRAINTS': 'TRUE' if self.config.enable_constitutional_constraints else 'FALSE'
            }
            dataset.SetMetadata(processing_metadata, 'PROCESSING')
            
            # Add quality metrics as metadata
            quality_metadata = {f"QUALITY_{k.upper()}": str(v) for k, v in quality_metrics.items()}
            dataset.SetMetadata(quality_metadata, 'QUALITY')
            
            # Add adaptive parameters as metadata
            adaptive_metadata = {f"ADAPTIVE_{k.upper()}": str(v) for k, v in adaptive_params.items()}
            dataset.SetMetadata(adaptive_metadata, 'ADAPTIVE')
            
            # Flush and close
            dataset.FlushCache()
            dataset = None
            
            self.logger.debug(f"Successfully saved enhanced results: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            raise
    
    def _flag_for_expert_review(self, filename: str, stats: Dict):
        """Flag file for expert review based on quality metrics."""
        if not self.expert_review:
            return
        
        try:
            review_data = {
                'filename': filename,
                'flagged_date': datetime.datetime.now().isoformat(),
                'composite_quality': stats.get('composite_quality', 0),
                'seafloor_type': stats.get('seafloor_type', 'unknown'),
                'ssim': stats.get('ssim', 0),
                'feature_preservation': stats.get('feature_preservation', 0),
                'consistency': stats.get('consistency', 0),
                'hydrographic_compliance': stats.get('hydrographic_compliance', 0),
                'flag_reason': 'Quality below threshold'
            }
            
            self.expert_review.flag_for_review(review_data)
            self.logger.info(f"Flagged {filename} for expert review (Quality: {stats.get('composite_quality', 0):.3f})")
            
        except Exception as e:
            self.logger.error(f"Error flagging {filename} for expert review: {e}")
    
    def _generate_comprehensive_reports(self):
        """Generate comprehensive processing reports."""
        try:
            # Save processing statistics
            stats_file = Path("processing_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(self.processing_stats, f, indent=2)
            
            self.logger.info(f"Processing statistics saved: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def _generate_enhanced_summary_report(self, successful_files: List[str], 
                                        failed_files: List[str], processing_stats: List[Dict]):
        """Generate enhanced summary report with detailed statistics."""
        try:
            # Calculate summary statistics
            if processing_stats:
                summary_stats = {
                    'avg_composite_quality': np.mean([s.get('composite_quality', 0) for s in processing_stats]),
                    'avg_ssim': np.mean([s.get('ssim', 0) for s in processing_stats]),
                    'avg_feature_preservation': np.mean([s.get('feature_preservation', 0) for s in processing_stats]),
                    'avg_consistency': np.mean([s.get('consistency', 0) for s in processing_stats]),
                    'avg_hydrographic_compliance': np.mean([s.get('hydrographic_compliance', 0) for s in processing_stats])
                }
                
                # Calculate seafloor type distribution
                seafloor_types = [s.get('seafloor_type', 'unknown') for s in processing_stats]
                seafloor_distribution = {st: seafloor_types.count(st) for st in set(seafloor_types)}
                
                # Calculate channel distribution
                channel_counts = [s.get('input_channels', 1) for s in processing_stats]
                channel_distribution = {
                    f"{channels}_channel": channel_counts.count(channels)
                    for channels in set(channel_counts)
                }
            else:
                summary_stats = {}
                seafloor_distribution = {}
                channel_distribution = {}
            
            # Create comprehensive report
            report = {
                'processing_date': datetime.datetime.now().isoformat(),
                'pipeline_version': 'Enhanced Bathymetric CAE v2.0',
                'tensorflow_version': self.tf_version,
                'configuration': asdict(self.config),
                'total_files': len(successful_files) + len(failed_files),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'success_rate': len(successful_files) / (len(successful_files) + len(failed_files)) * 100 if (len(successful_files) + len(failed_files)) > 0 else 0,
                'processing_time_total': self.processing_stats.get('total_processing_time', 0),
                'average_processing_time': self.processing_stats.get('total_processing_time', 0) / max(len(successful_files), 1),
                'summary_statistics': summary_stats,
                'seafloor_distribution': seafloor_distribution,
                'channel_distribution': channel_distribution,
                'successful_file_list': successful_files,
                'failed_file_list': failed_files,
                'detailed_stats': processing_stats,
                'enabled_features': {
                    'adaptive_processing': self.config.enable_adaptive_processing,
                    'expert_review': self.config.enable_expert_review,
                    'constitutional_constraints': self.config.enable_constitutional_constraints,
                    'ensemble_processing': True
                }
            }
            
            # Save comprehensive report
            report_file = Path("enhanced_processing_summary.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED PROCESSING SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Pipeline Version: Enhanced Bathymetric CAE v2.0")
            self.logger.info(f"TensorFlow Version: {self.tf_version}")
            self.logger.info(f"Processing Date: {report['processing_date']}")
            self.logger.info("-" * 80)
            self.logger.info("FILE PROCESSING STATISTICS:")
            self.logger.info(f"  Total files: {report['total_files']}")
            self.logger.info(f"  Successful: {report['successful_files']}")
            self.logger.info(f"  Failed: {report['failed_files']}")
            self.logger.info(f"  Success rate: {report['success_rate']:.1f}%")
            self.logger.info(f"  Total processing time: {report['processing_time_total']:.1f}s")
            self.logger.info(f"  Average time per file: {report['average_processing_time']:.1f}s")
            self.logger.info(f"  Processing rate: {(len(successful_files) / max(report['processing_time_total']/60, 0.001)):.1f} files/minute")
            self.logger.info("-" * 80)
            self.logger.info("ENABLED FEATURES:")
            for feature, enabled in report['enabled_features'].items():
                status = "[YES]" if enabled else "[NO]"
                feature_name = feature.replace('_', ' ').title()
                self.logger.info(f"  {status} {feature_name}")
            self.logger.info("=" * 80)
            self.logger.info(f"Detailed report saved to: {report_file.absolute()}")
            
            if failed_files:
                self.logger.warning(f"Failed files: {len(failed_files)}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced summary report: {e}")
    
    def _generate_expert_review_report(self):
        """Generate expert review report."""
        if not self.expert_review:
            return
        
        try:
            # Get pending reviews
            pending_reviews = self.expert_review.get_pending_reviews()
            review_stats = self.expert_review.get_review_statistics()
            
            # Generate review report
            review_report = {
                'generation_date': datetime.datetime.now().isoformat(),
                'pending_reviews': pending_reviews,
                'review_statistics': review_stats,
                'total_pending': len(pending_reviews)
            }
            
            # Save review report
            review_dir = Path("expert_reviews")
            review_dir.mkdir(exist_ok=True)
            review_file = review_dir / "pending_reviews.json"
            
            with open(review_file, 'w') as f:
                json.dump(review_report, f, indent=2)
            
            self.logger.info(f"Review statistics: {review_stats}")
            self.logger.info(f"Expert review report generated: {review_file}")
            self.logger.info(f"Files pending expert review: {len(pending_reviews)}")
            
        except Exception as e:
            self.logger.error(f"Error generating expert review report: {e}")
                