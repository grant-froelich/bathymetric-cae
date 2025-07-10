"""
Main processing pipeline for Enhanced Bathymetric CAE Processing.

This module contains the main pipeline orchestration logic that coordinates
all components including ensemble models, adaptive processing, and quality control.
"""

import gc
import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from ..config import Config
from ..core import (
    AdaptiveProcessor, BathymetricQualityMetrics, BathymetricConstraints,
    SeafloorClassifier, QualityLevel
)
from ..models import BathymetricEnsemble
from ..review import ExpertReviewSystem
from ..utils.logging_utils import get_performance_logger, get_progress_logger
from .data_processor import BathymetricProcessor
from .memory_utils import memory_monitor, cleanup_memory
from .training import ModelTrainer


class EnhancedBathymetricCAEPipeline:
    """Enhanced processing pipeline with all improvements."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_logger = get_performance_logger("pipeline")
        
        # Initialize components
        self.ensemble = BathymetricEnsemble(config)
        self.adaptive_processor = AdaptiveProcessor() if config.enable_adaptive_processing else None
        self.quality_metrics = BathymetricQualityMetrics()
        self.constraints = BathymetricConstraints() if config.enable_constitutional_constraints else None
        self.expert_review = ExpertReviewSystem() if config.enable_expert_review else None
        self.data_processor = BathymetricProcessor(config)
        self.model_trainer = ModelTrainer(config)
        
        # Processing statistics
        self.processing_stats = []
        self.successful_files = []
        self.failed_files = []
    
    def run(self, input_folder: str, output_folder: str, model_path: str):
        """Run the enhanced processing pipeline."""
        self.performance_logger.start_timer("total_pipeline")
        
        try:
            # Setup environment
            self._setup_environment()
            self._validate_paths(input_folder, output_folder)
            
            # Get file list
            file_list = self._get_valid_files(input_folder)
            if not file_list:
                raise ValueError(f"No valid files found in {input_folder}")
            
            self.logger.info(f"Found {len(file_list)} files to process")
            
            # Train or load ensemble
            with memory_monitor("model_setup"):
                ensemble_models = self._get_or_train_ensemble(model_path, file_list)
            
            # Process files with enhanced features
            with memory_monitor("file_processing"):
                self._process_files_enhanced(ensemble_models, file_list, output_folder)
            
            # Generate reports
            self._generate_comprehensive_reports()
            
            # Cleanup
            cleanup_memory()
            
            self.logger.info("Enhanced pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            total_time = self.performance_logger.end_timer("total_pipeline")
            self.logger.info(f"Total pipeline execution time: {total_time:.2f} seconds")
    
    def _setup_environment(self):
        """Setup processing environment."""
        from .memory_utils import optimize_gpu_memory
        
        # GPU optimization
        optimize_gpu_memory()
        
        # Create output directories
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        
        if self.expert_review:
            Path("expert_reviews").mkdir(exist_ok=True)
        
        self.logger.info("Environment setup completed")
    
    def _validate_paths(self, input_folder: str, output_folder: str):
        """Validate input and output paths."""
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_folder}")
        
        # Create output directory
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Validated paths - Input: {input_folder}, Output: {output_folder}")
    
    def _get_valid_files(self, input_folder: str) -> List[Path]:
        """Get list of valid files to process."""
        input_path = Path(input_folder)
        valid_files = []
        
        for ext in self.config.supported_formats:
            pattern = f"*{ext}"
            found_files = list(input_path.glob(pattern))
            valid_files.extend(found_files)
            
            # Also check subdirectories
            found_files_recursive = list(input_path.rglob(pattern))
            valid_files.extend([f for f in found_files_recursive if f not in valid_files])
        
        # Remove duplicates and sort
        valid_files = sorted(list(set(valid_files)))
        
        self.logger.info(f"Found {len(valid_files)} valid files")
        for ext in self.config.supported_formats:
            count = len([f for f in valid_files if f.suffix.lower() == ext])
            if count > 0:
                self.logger.info(f"  {ext}: {count} files")
        
        return valid_files
    
    def _get_or_train_ensemble(self, model_path: str, file_list: List[Path]):
        """Get existing ensemble or train new one."""
        model_base = Path(model_path).stem
        
        try:
            # Try to load existing ensemble
            self.ensemble.load_ensemble(model_base)
            self.logger.info(f"Loaded existing ensemble from {model_base}")
            return self.ensemble.models
            
        except Exception as e:
            self.logger.info(f"Creating new ensemble (could not load existing): {e}")
            return self._train_ensemble(model_base, file_list)
    
    def _train_ensemble(self, model_path: str, file_list: List[Path]):
        """Train new ensemble of models."""
        self.performance_logger.start_timer("ensemble_training")
        
        try:
            # Determine model parameters from sample file
            sample_file = file_list[0]
            input_data, _, _ = self.data_processor.preprocess_bathymetric_grid(sample_file)
            channels = input_data.shape[-1]
            
            self.logger.info(f"Training ensemble with {channels} input channels")
            
            # Create ensemble
            ensemble_models = self.ensemble.create_ensemble(channels)
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(file_list)
            
            # Train ensemble using ModelTrainer
            trained_models = self.model_trainer.train_ensemble(
                ensemble_models, X_train, y_train, model_path
            )
            
            # Update ensemble with trained models
            self.ensemble.models = trained_models
            
            # Save ensemble
            self.ensemble.save_ensemble(model_path)
            
            # Clean up training data
            del X_train, y_train
            gc.collect()
            
            self.logger.info("Ensemble training completed successfully")
            return trained_models
            
        finally:
            training_time = self.performance_logger.end_timer("ensemble_training")
            self.logger.info(f"Ensemble training time: {training_time:.2f} seconds")
    
    def _prepare_training_data(self, file_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare enhanced training data with adaptive preprocessing."""
        self.logger.info("Preparing training data...")
        progress_logger = get_progress_logger(len(file_list), "data_preparation")
        
        all_inputs = []
        all_targets = []
        
        for i, file_path in enumerate(file_list):
            try:
                progress_logger.update(1, f"Processing {file_path.name}")
                
                # Load and preprocess
                input_data, _, _ = self.data_processor.preprocess_bathymetric_grid(file_path)
                
                # Get adaptive parameters if enabled
                adaptive_params = {}
                if self.adaptive_processor:
                    depth_data = input_data[..., 0]
                    adaptive_params = self.adaptive_processor.get_processing_parameters(depth_data)
                
                # Apply adaptive preprocessing
                enhanced_input = self._apply_adaptive_preprocessing(input_data, adaptive_params)
                
                # Create training pair
                noisy_data = np.expand_dims(enhanced_input, axis=0)
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
    
    def _apply_adaptive_preprocessing(self, input_data: np.ndarray, 
                                    adaptive_params: Dict) -> np.ndarray:
        """Apply adaptive preprocessing based on seafloor type."""
        if not adaptive_params:
            return input_data
        
        from scipy import ndimage
        
        enhanced_data = input_data.copy()
        smoothing_factor = adaptive_params.get('smoothing_factor', 0.5)
        
        if smoothing_factor > 0:
            # Apply Gaussian smoothing
            sigma = smoothing_factor * 2
            for channel in range(enhanced_data.shape[-1]):
                enhanced_data[..., channel] = ndimage.gaussian_filter(
                    enhanced_data[..., channel], sigma=sigma
                )
        
        return enhanced_data
    
    def _process_files_enhanced(self, ensemble_models: List, file_list: List[Path], output_folder: str):
        """Process files using enhanced ensemble approach."""
        output_path = Path(output_folder)
        
        self.logger.info(f"Processing {len(file_list)} files with enhanced pipeline...")
        progress_logger = get_progress_logger(len(file_list), "file_processing")
        
        for i, file_path in enumerate(file_list, 1):
            try:
                progress_logger.update(1, f"Processing {file_path.name}")
                
                # Process single file with enhanced features
                stats = self._process_single_file_enhanced(file_path, output_path)
                self.processing_stats.append(stats)
                self.successful_files.append(file_path.name)
                
                # Check if flagging for expert review is needed
                if (self.expert_review and 
                    stats.get('composite_quality', 1.0) < self.config.quality_threshold):
                    self._flag_for_expert_review(file_path.name, stats)
                
                # Cleanup memory periodically
                if i % 5 == 0:
                    cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                self.failed_files.append(file_path.name)
                continue
    
    def _process_single_file_enhanced(self, file_path: Path, output_path: Path) -> Dict:
        """Process a single file with enhanced ensemble approach."""
        self.performance_logger.start_timer(f"process_{file_path.name}")
        
        try:
            # Load and preprocess data
            input_data, original_shape, geo_metadata = self.data_processor.preprocess_bathymetric_grid(file_path)
            
            # Get adaptive processing parameters
            adaptive_params = {}
            if self.adaptive_processor:
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
            from ..utils.visualization import create_enhanced_visualization
            create_enhanced_visualization(
                original_depth, final_prediction, uncertainty_data, 
                all_metrics, file_path, adaptive_params
            )
            
            # Return comprehensive statistics
            stats = {
                'filename': file_path.name,
                'processing_time': datetime.datetime.now().isoformat(),
                'seafloor_type': adaptive_params.get('seafloor_type', 'unknown'),
                'adaptive_params': adaptive_params,
                **all_metrics
            }
            
            self.logger.info(f"Enhanced processing complete for {file_path.name} - "
                           f"Quality: {all_metrics.get('composite_quality', 0):.3f}")
            
            return stats
            
        finally:
            processing_time = self.performance_logger.end_timer(f"process_{file_path.name}")
            self.logger.debug(f"File processing time: {processing_time:.2f} seconds")
    
    def _calculate_comprehensive_quality_metrics(self, original: np.ndarray, 
                                               cleaned: np.ndarray,
                                               uncertainty: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive quality metrics."""
        # Use the quality metrics calculator
        weights = self.config.get_quality_weights()
        return self.quality_metrics.calculate_all_metrics(original, cleaned, uncertainty, weights)
    
    def _save_enhanced_results(self, data: np.ndarray, output_path: Path, 
                             original_shape: Tuple[int, int], geo_metadata: Dict,
                             quality_metrics: Dict, adaptive_params: Dict):
        """Save enhanced results with comprehensive metadata."""
        # Delegate to data processor
        self.data_processor.save_enhanced_results(
            data, output_path, original_shape, geo_metadata, 
            quality_metrics, adaptive_params
        )
    
    def _flag_for_expert_review(self, filename: str, stats: Dict):
        """Flag file for expert review based on quality metrics."""
        if not self.expert_review:
            return
        
        try:
            quality_score = stats.get('composite_quality', 1.0)
            
            # Determine flag type based on low-quality metrics
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
    
    def _generate_comprehensive_reports(self):
        """Generate all reports and summaries."""
        try:
            # Generate enhanced summary report
            self._generate_enhanced_summary_report()
            
            # Generate expert review report
            if self.expert_review:
                self._generate_expert_review_report()
            
            # Generate performance report
            self._generate_performance_report()
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def _generate_enhanced_summary_report(self):
        """Generate comprehensive enhanced processing summary report."""
        report_path = Path("enhanced_processing_summary.json")
        
        # Calculate enhanced summary statistics
        if self.processing_stats:
            quality_scores = [s.get('composite_quality', 0) for s in self.processing_stats]
            ssim_scores = [s.get('ssim', 0) for s in self.processing_stats]
            feature_scores = [s.get('feature_preservation', 0) for s in self.processing_stats]
            
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
            seafloor_types = [s.get('seafloor_type', 'unknown') for s in self.processing_stats]
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
            'configuration': self.config.to_dict(),
            'total_files': len(self.successful_files) + len(self.failed_files),
            'successful_files': len(self.successful_files),
            'failed_files': len(self.failed_files),
            'success_rate': len(self.successful_files) / (len(self.successful_files) + len(self.failed_files)) * 100 if (len(self.successful_files) + len(self.failed_files)) > 0 else 0,
            'summary_statistics': summary_stats,
            'seafloor_distribution': seafloor_distribution,
            'successful_file_list': self.successful_files,
            'failed_file_list': self.failed_files,
            'detailed_stats': self.processing_stats,
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
        
        self._log_summary_statistics(report)
        
        self.logger.info(f"Enhanced summary report saved to: {report_path}")
    
    def _log_summary_statistics(self, report: Dict):
        """Log enhanced summary statistics."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("ENHANCED PROCESSING SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total files: {report['total_files']}")
        self.logger.info(f"Successful: {report['successful_files']}")
        self.logger.info(f"Failed: {report['failed_files']}")
        self.logger.info(f"Success rate: {report['success_rate']:.1f}%")
        
        summary_stats = report.get('summary_statistics', {})
        if summary_stats:
            self.logger.info(f"Mean composite quality: {summary_stats.get('mean_composite_quality', 0):.4f}")
            self.logger.info(f"High quality files (>0.8): {summary_stats.get('high_quality_files', 0)}")
            self.logger.info(f"Medium quality files (0.6-0.8): {summary_stats.get('medium_quality_files', 0)}")
            self.logger.info(f"Low quality files (<0.6): {summary_stats.get('low_quality_files', 0)}")
        
        seafloor_distribution = report.get('seafloor_distribution', {})
        if seafloor_distribution:
            self.logger.info("Seafloor type distribution:")
            for seafloor_type, count in seafloor_distribution.items():
                self.logger.info(f"  {seafloor_type}: {count}")
        
        self.logger.info(f"{'='*60}")
    
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
    
    def _generate_performance_report(self):
        """Generate performance report."""
        try:
            self.performance_logger.log_all_counters()
            
            performance_summary = {
                'generation_date': datetime.datetime.now().isoformat(),
                'total_files_processed': len(self.successful_files),
                'total_processing_time': self.performance_logger.timings.get('total_pipeline', 0),
                'average_time_per_file': (
                    self.performance_logger.timings.get('total_pipeline', 0) / 
                    max(len(self.successful_files), 1)
                ),
                'counters': self.performance_logger.counters.copy()
            }
            
            report_path = "logs/performance_summary.json"
            with open(report_path, 'w') as f:
                json.dump(performance_summary, f, indent=2)
            
            self.logger.info(f"Performance report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
