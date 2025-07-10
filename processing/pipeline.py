"""
Main processing pipeline for Enhanced Bathymetric CAE.
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
    """Enhanced processing pipeline with all improvements."""
    
    def __init__(self, config: Config):
        self.config = config
        self.ensemble = BathymetricEnsemble(config)
        self.adaptive_processor = AdaptiveProcessor()
        self.quality_metrics = BathymetricQualityMetrics()
        self.expert_review = ExpertReviewSystem() if config.enable_expert_review else None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self, input_folder: str, output_folder: str, model_path: str):
        """Run the enhanced processing pipeline."""
        try:
            # Setup
            self._setup_environment()
            self._validate_paths(input_folder, output_folder)
            
            # Get file list
            file_list = self._get_valid_files(input_folder)
            if not file_list:
                raise ValueError(f"No valid files found in {input_folder}")
            
            self.logger.info(f"Found {len(file_list)} files to process")
            
            # Train or load ensemble
            ensemble_models = self._get_or_train_ensemble(model_path, file_list)
            
            # Process files with enhanced features
            self._process_files_enhanced(ensemble_models, file_list, output_folder)
            
            # Generate expert review report
            if self.expert_review:
                self._generate_expert_review_report()
            
            self.logger.info("Enhanced pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _setup_environment(self):
        """Setup processing environment."""
        optimize_gpu_memory()
        
        # Create output directories
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        Path("expert_reviews").mkdir(exist_ok=True)
    
    def _validate_paths(self, input_folder: str, output_folder: str):
        """Validate input and output paths."""
        if not Path(input_folder).exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    def _get_valid_files(self, input_folder: str) -> List[Path]:
        """Get list of valid files to process."""
        input_path = Path(input_folder)
        valid_files = []
        
        for ext in self.config.supported_formats:
            valid_files.extend(input_path.glob(f"*{ext}"))
        
        return sorted(valid_files)
    
    def _get_or_train_ensemble(self, model_path: str, file_list: List[Path]):
        """Get existing ensemble or train new one."""
        try:
            # Try to load existing ensemble
            from models.ensemble import BathymetricEnsemble
            ensemble_models = []
            for i in range(self.config.ensemble_size):
                model_file = f"{model_path}_ensemble_{i}.h5"
                model = load_model(model_file, compile=False)
                ensemble_models.append(model)
            
            self.ensemble.models = ensemble_models
            self.logger.info(f"Loaded existing ensemble of {len(ensemble_models)} models")
            return ensemble_models
            
        except Exception as e:
            self.logger.info(f"Creating new ensemble (could not load existing): {e}")
            return self._train_ensemble(model_path, file_list)
    
    def _train_ensemble(self, model_path: str, file_list: List[Path]):
        """Train new ensemble of models."""
        # Determine model parameters from sample file
        sample_file = file_list[0]
        input_data, _, _ = self._preprocess_sample_file(sample_file)
        channels = input_data.shape[-1]
        
        # Create ensemble
        ensemble_models = self.ensemble.create_ensemble(channels)
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(file_list)
        
        # Train each model in ensemble
        for i, model in enumerate(ensemble_models):
            self.logger.info(f"Training ensemble model {i+1}/{len(ensemble_models)}")
            
            # Setup callbacks for this model
            model_callbacks = self._setup_callbacks(f"{model_path}_ensemble_{i}.h5")
            
            # Train model
            with memory_monitor(f"Training ensemble model {i+1}"):
                history = model.fit(
                    X_train, y_train,
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    validation_split=self.config.validation_split,
                    callbacks=model_callbacks,
                    verbose=1
                )
            
            # Plot training history for this model
            plot_training_history(history, f"training_history_ensemble_{i}.png")
        
        # Clean up training data
        del X_train, y_train
        gc.collect()
        
        return ensemble_models
    
    def _preprocess_sample_file(self, file_path: Path):
        """Preprocess a sample file to determine parameters."""
        processor = BathymetricProcessor(self.config)
        return processor.preprocess_bathymetric_grid(file_path)
    
    def _prepare_training_data(self, file_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare enhanced training data with adaptive preprocessing."""
        all_inputs = []
        all_targets = []
    
        processor = BathymetricProcessor(self.config)
    
        for i, file_path in enumerate(file_list):
            try:
                input_data, _, _ = processor.preprocess_bathymetric_grid(file_path)
            
                # Get adaptive parameters for this data
                depth_data = input_data[..., 0]
                adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
            
                # Apply adaptive preprocessing
                enhanced_input = self._apply_adaptive_preprocessing(input_data, adaptive_params)
            
                noisy_data = np.expand_dims(enhanced_input, axis=0)
                clean_data = noisy_data[..., :1]  # Use first channel as target
            
                all_inputs.append(noisy_data)
                all_targets.append(clean_data)
            
                # ✅ MEMORY MANAGEMENT FIXES:
                # Clear intermediate variables
                del input_data, depth_data, enhanced_input, noisy_data, clean_data
            
                # Periodic garbage collection every 5 files
                if (i + 1) % 5 == 0:
                    import gc
                    gc.collect()
                    self.logger.debug(f"Memory cleanup after processing {i + 1} files")
            
                # Memory usage warning for large datasets
                if len(all_inputs) > 50:
                    self.logger.warning(f"Loading {len(all_inputs)} files in memory - consider batch processing")
            
                self.logger.debug(f"Loaded training data from {file_path.name}")
            
            except Exception as e:
                self.logger.warning(f"Skipping {file_path.name}: {e}")
                continue
    
        if not all_inputs:
            raise ValueError("No valid training data found")
    
        # ✅ Final memory optimization
        X_train = np.vstack(all_inputs).astype(np.float32)
        y_train = np.vstack(all_targets).astype(np.float32)
    
        # Clear the lists immediately after stacking
        del all_inputs, all_targets
        import gc
        gc.collect()
    
        self.logger.info(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
    
        return X_train, y_train
    
    def _apply_adaptive_preprocessing(self, input_data: np.ndarray, 
                                    adaptive_params: Dict) -> np.ndarray:
        """Apply adaptive preprocessing based on seafloor type."""
        # Apply smoothing based on seafloor type
        smoothing_factor = adaptive_params.get('smoothing_factor', 0.5)
        
        if smoothing_factor > 0:
            # Apply Gaussian smoothing
            sigma = smoothing_factor * 2
            for channel in range(input_data.shape[-1]):
                input_data[..., channel] = ndimage.gaussian_filter(
                    input_data[..., channel], sigma=sigma
                )
        
        return input_data
    
    def _setup_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """Setup training callbacks."""
        callback_list = [
            callbacks.TensorBoard(
                log_dir=self.config.log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(f'{model_path}_training_log.csv')
        ]
        
        return callback_list
    
    def _process_files_enhanced(self, ensemble_models: List, file_list: List[Path], output_folder: str):
        """Process files using enhanced ensemble approach."""
        output_path = Path(output_folder)
        processor = BathymetricProcessor(self.config)
    
        successful_files = []
        failed_files = []
        processing_stats = []
    
        self.logger.info(f"Processing {len(file_list)} files with enhanced pipeline...")
    
        for i, file_path in enumerate(file_list, 1):
            try:
                self.logger.info(f"Processing file {i}/{len(file_list)}: {file_path.name}")
            
                # Process single file with enhanced features
                stats = self._process_single_file_enhanced(
                    processor, file_path, output_path
                )
                processing_stats.append(stats)
                successful_files.append(file_path.name)
            
                # Check if flagging for expert review is needed
                if (self.expert_review and 
                    stats.get('composite_quality', 1.0) < self.config.quality_threshold):
                    self._flag_for_expert_review(file_path.name, stats)
            
                # ✅ MEMORY MANAGEMENT FIXES:
                # Clear TensorFlow session periodically
                if i % 10 == 0:
                    import gc
                    gc.collect()
                    # Clear TensorFlow backend
                    tf.keras.backend.clear_session()
                    self.logger.info(f"Memory cleanup after processing {i} files")
            
                # Force garbage collection every 5 files
                if i % 5 == 0:
                    import gc
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                failed_files.append(file_path.name)
            
                # ✅ Cleanup on error too
                import gc
                gc.collect()
                continue
    
        # ✅ Final cleanup
        import gc
        gc.collect()
        tf.keras.backend.clear_session()
    
        # Generate enhanced summary report
        self._generate_enhanced_summary_report(successful_files, failed_files, processing_stats)
    
    def _process_single_file_enhanced(self, processor, file_path: Path, output_path: Path) -> Dict:
        """Process a single file with enhanced ensemble approach."""
        log_memory_usage(f"Starting {file_path.name}")
        
        with memory_monitor(f"Enhanced processing {file_path.name}"):
            # Load and preprocess data
            input_data, original_shape, geo_metadata = processor.preprocess_bathymetric_grid(file_path)
            
            # Get adaptive processing parameters
            depth_data = input_data[..., 0]
            adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
            
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
            create_enhanced_visualization(
                original_depth, final_prediction, uncertainty_data, 
                all_metrics, file_path, adaptive_params
            )
            
            # Return comprehensive statistics
            stats = {
                'filename': file_path.name,
                'processing_time': datetime.datetime.now().isoformat(),
                'seafloor_type': self.adaptive_processor.seafloor_classifier.classify(depth_data).value,
                'adaptive_params': adaptive_params,
                **all_metrics
            }
            
            self.logger.info(f"Enhanced processing complete for {file_path.name} - "
                           f"Quality: {all_metrics.get('composite_quality', 0):.3f}")
            
            log_memory_usage(f"Completed {file_path.name}")
            return stats
    
    def _calculate_comprehensive_quality_metrics(self, original: np.ndarray, 
                                               cleaned: np.ndarray,
                                               uncertainty: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive quality metrics."""
        metrics = {}
        
        # Standard metrics
        from skimage.metrics import structural_similarity as ssim
        metrics['ssim'] = self._calculate_ssim_safe(original, cleaned)
        metrics['mae'] = float(np.mean(np.abs(original - cleaned)))
        metrics['rmse'] = float(np.sqrt(np.mean((original - cleaned)**2)))
        
        # Domain-specific metrics
        metrics['roughness'] = self.quality_metrics.calculate_roughness(cleaned)
        metrics['feature_preservation'] = self.quality_metrics.calculate_feature_preservation(original, cleaned)
        metrics['consistency'] = self.quality_metrics.calculate_depth_consistency(cleaned)
        metrics['hydrographic_compliance'] = self.quality_metrics.calculate_hydrographic_standards_compliance(
            cleaned, uncertainty
        )
        
        # Uncertainty metrics
        if uncertainty is not None:
            metrics.update(self._calculate_uncertainty_metrics(uncertainty, cleaned))
        
        # Calculate composite quality score
        metrics['composite_quality'] = (
            self.config.ssim_weight * metrics['ssim'] +
            self.config.roughness_weight * (1.0 - min(metrics['roughness'], 1.0)) +
            self.config.feature_preservation_weight * metrics['feature_preservation'] +
            self.config.consistency_weight * metrics['consistency']
        )
        
        return metrics
    
    def _calculate_ssim_safe(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            from skimage.metrics import structural_similarity as ssim
            if original.shape != cleaned.shape:
                return 0.0
            
            data_range = float(max(cleaned.max() - cleaned.min(), 1e-8))
            
            return ssim(
                original.astype(np.float64),
                cleaned.astype(np.float64),
                data_range=data_range,
                gaussian_weights=True,
                win_size=min(7, min(original.shape[-2:]))
            )
        except Exception as e:
            self.logger.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    def _calculate_uncertainty_metrics(self, uncertainty_data: np.ndarray, 
                                     cleaned_data: np.ndarray) -> Dict:
        """Calculate comprehensive uncertainty metrics."""
        try:
            return {
                'mean_uncertainty': float(np.mean(uncertainty_data)),
                'std_uncertainty': float(np.std(uncertainty_data)),
                'max_uncertainty': float(np.max(uncertainty_data)),
                'uncertainty_reduction': float(np.mean(uncertainty_data) - np.mean(np.abs(uncertainty_data - cleaned_data))),
                'uncertainty_correlation': float(np.corrcoef(uncertainty_data.flatten(), cleaned_data.flatten())[0, 1])
            }
        except Exception as e:
            self.logger.error(f"Error calculating uncertainty metrics: {e}")
            return {}
    
    def _save_enhanced_results(self, data: np.ndarray, output_path: Path, 
                             original_shape: Tuple[int, int], geo_metadata: Dict,
                             quality_metrics: Dict, adaptive_params: Dict):
        """Save enhanced results with comprehensive metadata."""
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
            if 'geotransform' in geo_metadata:
                dataset.SetGeoTransform(geo_metadata['geotransform'])
            if 'projection' in geo_metadata:
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
            
            self.logger.info(f"Successfully saved enhanced results: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            raise
    
    def _flag_for_expert_review(self, filename: str, stats: Dict):
        """Flag file for expert review based on quality metrics."""
        if not self.expert_review:
            return
        
        try:
            # Determine flag type based on low-quality metrics
            quality_score = stats.get('composite_quality', 1.0)
            flag_type = "low_quality"
            
            if stats.get('feature_preservation', 1.0) < 0.5:
                flag_type = "feature_loss"
            elif stats.get('hydrographic_compliance', 1.0) < 0.7:
                flag_type = "standards_violation"
            elif stats.get('ssim', 1.0) < 0.6:
                flag_type = "poor_similarity"
            
            # Flag entire file (simplified - could be region-specific)
            self.expert_review.flag_for_review(
                filename, (0, 0, 512, 512), flag_type, 1.0 - quality_score
            )
            
            self.logger.warning(f"Flagged {filename} for expert review: {flag_type} "
                              f"(quality: {quality_score:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error flagging {filename} for review: {e}")
    
    def _generate_expert_review_report(self):
        """Generate expert review report."""
        if not self.expert_review:
            return
        
        try:
            pending_reviews = self.expert_review.get_pending_reviews()
            
            report = {
                'generation_date': datetime.datetime.now().isoformat(),
                'total_pending_reviews': len(pending_reviews),
                'pending_reviews': pending_reviews
            }
            
            report_path = "expert_reviews/pending_reviews.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Expert review report generated: {report_path}")
            self.logger.info(f"Files pending expert review: {len(pending_reviews)}")
            
        except Exception as e:
            self.logger.error(f"Error generating expert review report: {e}")
    
    def _generate_enhanced_summary_report(self, successful_files: List[str], 
                                        failed_files: List[str], 
                                        processing_stats: List[Dict]):
        """Generate comprehensive enhanced processing summary report."""
        report_path = Path("enhanced_processing_summary.json")
        
        # Calculate enhanced summary statistics
        if processing_stats:
            quality_scores = [s.get('composite_quality', 0) for s in processing_stats]
            ssim_scores = [s.get('ssim', 0) for s in processing_stats]
            feature_scores = [s.get('feature_preservation', 0) for s in processing_stats]
            
            summary_stats = {
                'mean_composite_quality': float(np.mean(quality_scores)),
                'std_composite_quality': float(np.std(quality_scores)),
                'min_composite_quality': float(np.min(quality_scores)),
                'max_composite_quality': float(np.max(quality_scores)),
                'mean_ssim': float(np.mean(ssim_scores)),
                'mean_feature_preservation': float(np.mean(feature_scores)),
                'high_quality_files': len([q for q in quality_scores if q > 0.8]),
                'medium_quality_files': len([q for q in quality_scores if 0.6 <= q <= 0.8]),
                'low_quality_files': len([q for q in quality_scores if q < 0.6])
            }
            
            # Seafloor type distribution
            seafloor_types = [s.get('seafloor_type', 'unknown') for s in processing_stats]
            seafloor_distribution = {
                seafloor_type: seafloor_types.count(seafloor_type) 
                for seafloor_type in set(seafloor_types)
            }
        else:
            summary_stats = {}
            seafloor_distribution = {}
        
        # Create comprehensive report
        report = {
            'processing_date': datetime.datetime.now().isoformat(),
            'pipeline_version': 'Enhanced Bathymetric CAE v2.0',
            'configuration': asdict(self.config),
            'total_files': len(successful_files) + len(failed_files),
            'successful_files': len(successful_files),
            'failed_files': len(failed_files),
            'success_rate': len(successful_files) / (len(successful_files) + len(failed_files)) * 100 if (len(successful_files) + len(failed_files)) > 0 else 0,
            'summary_statistics': summary_stats,
            'seafloor_distribution': seafloor_distribution,
            'successful_file_list': successful_files,
            'failed_file_list': failed_files,
            'detailed_stats': processing_stats,
            'features_enabled': {
                'adaptive_processing': self.config.enable_adaptive_processing,
                'expert_review': self.config.enable_expert_review,
                'constitutional_constraints': self.config.enable_constitutional_constraints,
                'ensemble_processing': True
            }
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log enhanced summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("ENHANCED PROCESSING SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total files: {report['total_files']}")
        self.logger.info(f"Successful: {report['successful_files']}")
        self.logger.info(f"Failed: {report['failed_files']}")
        self.logger.info(f"Success rate: {report['success_rate']:.1f}%")
        
        if summary_stats:
            self.logger.info(f"Mean composite quality: {summary_stats['mean_composite_quality']:.4f}")
            self.logger.info(f"High quality files (>0.8): {summary_stats['high_quality_files']}")
            self.logger.info(f"Medium quality files (0.6-0.8): {summary_stats['medium_quality_files']}")
            self.logger.info(f"Low quality files (<0.6): {summary_stats['low_quality_files']}")
        
        if seafloor_distribution:
            self.logger.info("Seafloor type distribution:")
            for seafloor_type, count in seafloor_distribution.items():
                self.logger.info(f"  {seafloor_type}: {count}")
        
        self.logger.info(f"Detailed report saved to: {report_path}")
        self.logger.info(f"{'='*60}")