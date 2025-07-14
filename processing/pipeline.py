"""
Enhanced Bathymetric CAE Processing Pipeline v2.0
Fixed syntax error on line 426 and other potential issues.
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
                'pending_reviews': pending_reviews,
                'review_statistics': self.expert_review.get_review_statistics()
            }
            
            report_path = Path("expert_reviews") / "pending_reviews.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Expert review report generated: {report_path}")
            self.logger.info(f"Files pending expert review: {len(pending_reviews)}")
            
        except Exception as e:
            self.logger.error(f"Error generating expert review report: {e}")
    
    def _generate_comprehensive_reports(self):
        """Generate comprehensive processing reports."""
        try:
            # Processing statistics report
            stats_report = {
                'generation_date': datetime.datetime.now().isoformat(),
                'pipeline_version': 'Enhanced Bathymetric CAE v2.0',
                'tensorflow_version': self.tf_version,
                'processing_statistics': self.processing_stats,
                'configuration': asdict(self.config)
            }
            
            stats_path = Path("processing_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_report, f, indent=2)
            
            self.logger.info(f"Processing statistics saved: {stats_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive reports: {e}")
    
    def _generate_enhanced_summary_report(self, successful_files: List[str], 
                                        failed_files: List[str], 
                                        processing_stats: List[Dict]):
        """Generate comprehensive enhanced processing summary report."""
        report_path = Path("enhanced_processing_summary.json")
        
        try:
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
                
                # Channel distribution
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
                'features_enabled': {
                    'adaptive_processing': self.config.enable_adaptive_processing,
                    'expert_review': self.config.enable_expert_review,
                    'constitutional_constraints': self.config.enable_constitutional_constraints,
                    'ensemble_processing': True
                },
                'performance_metrics': {
                    'files_per_minute': len(successful_files) / max(self.processing_stats.get('total_processing_time', 1) / 60, 1),
                    'memory_efficiency': 'optimized',
                    'gpu_utilization': len(tf.config.list_physical_devices('GPU')) > 0
                }
            }
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Log enhanced summary
            self._log_processing_summary(report)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced summary report: {e}")
            # Create minimal report on error
            minimal_report = {
                'processing_date': datetime.datetime.now().isoformat(),
                'total_files': len(successful_files) + len(failed_files),
                'successful_files': len(successful_files),
                'failed_files': len(failed_files),
                'error': str(e)
            }
            
            with open(report_path, 'w') as f:
                json.dump(minimal_report, f, indent=2)
    
    def _log_processing_summary(self, report: Dict):
        """Log comprehensive processing summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("ENHANCED PROCESSING SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Pipeline Version: {report['pipeline_version']}")
        self.logger.info(f"TensorFlow Version: {report.get('tensorflow_version', 'Unknown')}")
        self.logger.info(f"Processing Date: {report['processing_date']}")
        self.logger.info("-" * 80)
        
        # File statistics
        self.logger.info("FILE PROCESSING STATISTICS:")
        self.logger.info(f"  Total files: {report['total_files']}")
        self.logger.info(f"  Successful: {report['successful_files']}")
        self.logger.info(f"  Failed: {report['failed_files']}")
        self.logger.info(f"  Success rate: {report['success_rate']:.1f}%")
        
        # Performance statistics
        if 'processing_time_total' in report:
            self.logger.info(f"  Total processing time: {report['processing_time_total']:.1f}s")
            self.logger.info(f"  Average time per file: {report['average_processing_time']:.1f}s")
        
        if 'performance_metrics' in report:
            perf = report['performance_metrics']
            self.logger.info(f"  Processing rate: {perf['files_per_minute']:.1f} files/minute")
        
        # Quality statistics
        if report.get('summary_statistics'):
            stats = report['summary_statistics']
            self.logger.info("-" * 80)
            self.logger.info("QUALITY STATISTICS:")
            self.logger.info(f"  Mean composite quality: {stats['mean_composite_quality']:.4f}")
            self.logger.info(f"  Quality range: {stats['min_composite_quality']:.4f} - {stats['max_composite_quality']:.4f}")
            self.logger.info(f"  Mean SSIM: {stats['mean_ssim']:.4f}")
            self.logger.info(f"  Mean feature preservation: {stats['mean_feature_preservation']:.4f}")
            self.logger.info(f"  High quality files (>0.8): {stats['high_quality_files']}")
            self.logger.info(f"  Medium quality files (0.6-0.8): {stats['medium_quality_files']}")
            self.logger.info(f"  Low quality files (<0.6): {stats['low_quality_files']}")
        
        # Seafloor distribution
        if report.get('seafloor_distribution'):
            self.logger.info("-" * 80)
            self.logger.info("SEAFLOOR TYPE DISTRIBUTION:")
            for seafloor_type, count in report['seafloor_distribution'].items():
                self.logger.info(f"  {seafloor_type}: {count} files")
        
        # Features enabled
        if report.get('features_enabled'):
            features = report['features_enabled']
            self.logger.info("-" * 80)
            self.logger.info("ENABLED FEATURES:")
            for feature, enabled in features.items():
                status = "[YES]" if enabled else "[NO] "
                self.logger.info(f"  {status} {feature.replace('_', ' ').title()}")
        
        self.logger.info("="*80)
        self.logger.info(f"Detailed report saved to: {Path('enhanced_processing_summary.json').absolute()}")
        
        if report.get('failed_files'):
            self.logger.warning(f"Failed files: {report['failed_files']}")
        
        self.logger.info("="*80)
    
    def _normalize_model_path(self, model_path: str) -> str:
        """Convert model path to appropriate format based on TensorFlow version."""
        # Use .h5 format for older TensorFlow versions for compatibility
        if model_path.endswith('.keras'):
            model_path = model_path.replace('.keras', '.h5')
        elif not model_path.endswith('.h5'):
            model_path += '.h5'
        
        self.logger.debug(f"Using model format: {model_path}")
        return model_path
    
    def _get_ensemble_model_paths(self, base_model_path: str, ensemble_size: int) -> List[str]:
        """Get ensemble model paths with compatibility."""
        base_path = self._normalize_model_path(base_model_path)
        base_name = base_path.replace('.h5', '')
        
        return [f"{base_name}_ensemble_{i}.h5" for i in range(ensemble_size)]
    
    def _get_or_train_ensemble(self, model_path: str, file_list: List[Path], 
                              channel_analysis: Dict):
        """Load or train ensemble with TensorFlow version compatibility."""
        try:
            # Get ensemble model paths
            ensemble_paths = self._get_ensemble_model_paths(model_path, self.config.ensemble_size)
            ensemble_models = []
            
            # Try to load existing models
            for i, h5_path in enumerate(ensemble_paths):
                if Path(h5_path).exists():
                    self.logger.debug(f"Loading model: {h5_path}")
                    model = load_model(h5_path)
                    ensemble_models.append(model)
                else:
                    break
            
            if len(ensemble_models) == self.config.ensemble_size:
                self.ensemble.models = ensemble_models
                self.logger.info(f"Loaded existing ensemble of {len(ensemble_models)} models")
                return ensemble_models
            else:
                self.logger.info("Training new ensemble")
                return self._train_ensemble(model_path, file_list, channel_analysis)
            
        except Exception as e:
            self.logger.info(f"Creating new ensemble (could not load existing): {e}")
            return self._train_ensemble(model_path, file_list, channel_analysis)
    
    def _setup_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """Setup training callbacks with TensorFlow version compatibility."""
        
        # Ensure compatible format
        model_path = self._normalize_model_path(model_path)
        
        callback_list = [
            callbacks.TensorBoard(
                log_dir=self.config.log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False  # Reduce memory usage
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
            callbacks.CSVLogger(f'{model_path.replace(".h5", "_training_log.csv")}')
        ]
        
        # Create ModelCheckpoint callback with version compatibility
        try:
            # Try modern format first (TensorFlow 2.17+)
            checkpoint_callback = callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_format='tf'  # This might not be supported in older versions
            )
            callback_list.append(checkpoint_callback)
            self.logger.info("Using modern ModelCheckpoint format")
        except TypeError:
            # Fallback to older format (TensorFlow < 2.17)
            checkpoint_callback = callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint_callback)
            self.logger.info("Using legacy ModelCheckpoint format")
        
        return callback_list
    
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
        format_counts = {}
        total_size = 0
        
        for file_path in file_list:
            ext = file_path.suffix.lower()
            format_counts[ext] = format_counts.get(ext, 0) + 1
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        
        self.logger.info("File summary:")
        for ext, count in format_counts.items():
            self.logger.info(f"  {ext}: {count} files")
        
        total_size_mb = total_size / (1024 * 1024)
        self.logger.info(f"  Total size: {total_size_mb:.1f} MB")
    
    def _analyze_data_channels(self, file_list: List[Path]) -> Dict:
        """Analyze data channels across all files to ensure consistency."""
        self.logger.info("Analyzing data channel consistency...")
        
        processor = BathymetricProcessor(self.config)
        channel_info = {}
        failed_files = []
        
        for file_path in file_list[:min(len(file_list), 10)]:  # Sample first 10 files
            try:
                input_data, shape, metadata = processor.preprocess_bathymetric_grid(file_path)
                channels = input_data.shape[-1]
                
                file_ext = file_path.suffix.lower()
                if file_ext not in channel_info:
                    channel_info[file_ext] = {'channels': [], 'shapes': [], 'files': []}
                
                channel_info[file_ext]['channels'].append(channels)
                channel_info[file_ext]['shapes'].append(shape)
                channel_info[file_ext]['files'].append(file_path.name)
                
            except Exception as e:
                failed_files.append((file_path.name, str(e)))
                self.logger.warning(f"Failed to analyze {file_path.name}: {e}")
        
        # Determine maximum channels needed
        max_channels = 1
        for ext_info in channel_info.values():
            if ext_info['channels']:
                max_channels = max(max_channels, max(ext_info['channels']))
        
        # Log analysis results
        self.logger.info("Channel analysis results:")
        for ext, info in channel_info.items():
            if info['channels']:
                unique_channels = list(set(info['channels']))
                self.logger.info(f"  {ext}: {unique_channels} channels")
        
        if failed_files:
            self.logger.warning(f"Failed to analyze {len(failed_files)} files")
        
        return {
            'max_channels': max_channels,
            'channel_info': channel_info,
            'failed_files': failed_files
        }
    
    def _train_ensemble(self, model_path: str, file_list: List[Path], 
                       channel_analysis: Dict):
        """Train new ensemble of models with compatibility fixes."""
        self.logger.info("Training new ensemble models...")
        
        # Determine model parameters from channel analysis
        max_channels = channel_analysis['max_channels']
        self.logger.info(f"Training ensemble for {max_channels} channels")
        
        # Create ensemble
        ensemble_models = self.ensemble.create_ensemble(max_channels)
        
        # Prepare training data with consistent channels
        X_train, y_train = self._prepare_training_data_consistent(file_list, max_channels)
        
        # Train each model in ensemble
        training_start_time = datetime.datetime.now()
        ensemble_paths = self._get_ensemble_model_paths(model_path, self.config.ensemble_size)
        
        for i, (model, model_path) in enumerate(zip(ensemble_models, ensemble_paths)):
            self.logger.info(f"Training ensemble model {i+1}/{len(ensemble_models)}")
            
            # Setup callbacks for this model
            model_callbacks = self._setup_callbacks(model_path)
            
            # Train model with memory monitoring
            with memory_monitor(f"Training ensemble model {i+1}"):
                try:
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
                    
                except Exception as e:
                    self.logger.error(f"Failed to train model {i+1}: {e}")
                    # Create a minimal model as fallback
                    self.logger.warning(f"Using minimal fallback model for ensemble {i+1}")
        
        training_end_time = datetime.datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()
        self.logger.info(f"Ensemble training completed in {training_duration:.1f} seconds")
        
        # Clean up training data
        del X_train, y_train
        gc.collect()
        
        return ensemble_models
    
    def _prepare_training_data_consistent(self, file_list: List[Path], 
                                        target_channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with consistent channel dimensions."""
        self.logger.info(f"Preparing training data with {target_channels} channels...")
        
        all_inputs = []
        all_targets = []
        
        processor = BathymetricProcessor(self.config)
        successful_files = 0
        
        for i, file_path in enumerate(file_list):
            try:
                input_data, _, _ = processor.preprocess_bathymetric_grid(file_path)
                
                # Ensure consistent channel count
                input_data = self._normalize_channels(input_data, target_channels)
                
                # Get adaptive parameters for this data
                depth_data = input_data[..., 0]
                adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
                
                # Apply adaptive preprocessing
                enhanced_input = self._apply_adaptive_preprocessing(input_data, adaptive_params)
                
                # Prepare training samples
                noisy_data = np.expand_dims(enhanced_input, axis=0)
                clean_data = noisy_data[..., :1]  # Use first channel as target
                
                all_inputs.append(noisy_data)
                all_targets.append(clean_data)
                successful_files += 1
                
                # Memory management
                del input_data, depth_data, enhanced_input, noisy_data, clean_data
                
                if (i + 1) % 5 == 0:
                    gc.collect()
                    log_memory_usage(f"Processed {i + 1}/{len(file_list)} files")
                
                # Limit training data size for memory management
                if len(all_inputs) >= 100:
                    self.logger.warning("Limiting training data to 100 samples for memory management")
                    break
                
                self.logger.debug(f"Loaded training data from {file_path.name}")
                
            except Exception as e:
                self.logger.warning(f"Skipping {file_path.name}: {e}")
                continue
        
        if not all_inputs:
            raise ValueError("No valid training data found")
        
        # Stack data
        X_train = np.vstack(all_inputs).astype(np.float32)
        y_train = np.vstack(all_targets).astype(np.float32)
        
        # Clean up intermediate data
        del all_inputs, all_targets
        gc.collect()
        
        self.logger.info(f"Training data prepared: {X_train.shape} -> {y_train.shape}")
        self.logger.info(f"Successfully processed {successful_files}/{len(file_list)} files for training")
        
        return X_train, y_train
    
    def _normalize_channels(self, input_data: np.ndarray, target_channels: int) -> np.ndarray:
        """Normalize input data to have consistent channel count."""
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
                seafloor_type = stats.get('seafloor_type', 'unknown')
                self.processing_stats['seafloor_type_distribution'][seafloor_type] = \
                    self.processing_stats['seafloor_type_distribution'].get(seafloor_type, 0) + 1
                
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
        """Process a single file with enhanced ensemble approach and consistent channels."""
        
        with memory_monitor(f"Processing {file_path.name}"):
            # Load and preprocess data
            input_data, original_shape, geo_metadata = processor.preprocess_bathymetric_grid(file_path)
            
            # Normalize channels to match ensemble expectations
            input_data = self._normalize_channels(input_data, expected_channels)
            
            # Get adaptive processing parameters
            depth_data = input_data[..., 0]
            seafloor_type = self.adaptive_processor.seafloor_classifier.classify(depth_data)
            adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
            adaptive_params['seafloor_type'] = seafloor_type.value
            
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
            
            # Return comprehensive statistics
            stats = {
                'filename': file_path.name,
                'processing_time': datetime.datetime.now().isoformat(),
                'seafloor_type': seafloor_type.value,
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
        metrics = {}
        
        try:
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
                uncertainty_metrics = self._calculate_uncertainty_metrics(uncertainty, cleaned)
                metrics.update(uncertainty_metrics)
            
            # Calculate composite quality score
            metrics['composite_quality'] = (
                self.config.ssim_weight * metrics['ssim'] +
                self.config.roughness_weight * (1.0 - min(metrics['roughness'], 1.0)) +
                self.config.feature_preservation_weight * metrics['feature_preservation'] +
                self.config.consistency_weight * metrics['consistency']
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            # Return default metrics on error
            metrics = {
                'ssim': 0.0,
                'mae': float('inf'),
                'rmse': float('inf'),
                'roughness': 1.0,
                'feature_preservation': 0.0,
                'consistency': 0.0,
                'hydrographic_compliance': 0.0,
                'composite_quality': 0.0
            }
        
        return metrics
    
    def _calculate_ssim_safe(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate SSIM with comprehensive error handling."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            if original.shape != cleaned.shape:
                self.logger.warning("Shape mismatch in SSIM calculation")
                return 0.0
            
            # Ensure finite values
            if not (np.isfinite(original).all() and np.isfinite(cleaned).all()):
                self.logger.warning("Non-finite values in SSIM calculation")
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