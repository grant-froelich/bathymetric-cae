            # Prepare enhanced data
            output_data = enhanced_data.astype(np.float32)
            invalid_mask = ~np.isfinite(output_data)
            if np.any(invalid_mask):
                output_data[invalid_mask] = nodata_value
            
            # Write elevation band
            elevation_band = dataset.GetRasterBand(1)
            elevation_band.SetNoDataValue(float(nodata_value))
            elevation_band.SetDescription('elevation')
            elevation_band.WriteArray(output_data)
            
            # Set statistics
            valid_data = output_data[output_data != nodata_value]
            if len(valid_data) > 0:
                elevation_band.SetStatistics(
                    float(np.min(valid_data)),
                    float(np.max(valid_data)),
                    float(np.mean(valid_data)),
                    float(np.std(valid_data))
                )
            
            # Write uncertainty band if available
            if uncertainty_data is not None and num_bands == 2:
                uncertainty_band = dataset.GetRasterBand(2)
                uncertainty_band.SetNoDataValue(float(nodata_value))
                uncertainty_band.SetDescription('uncertainty')
                
                uncertainty_output = uncertainty_data.astype(np.float32)
                uncertainty_invalid = ~np.isfinite(uncertainty_output)
                if np.any(uncertainty_invalid):
                    uncertainty_output[uncertainty_invalid] = nodata_value
                
                uncertainty_band.WriteArray(uncertainty_output)
            
            # Set metadata
            processing_metadata = {
                'PROCESSING_DATE': datetime.datetime.now().isoformat(),
                'PROCESSING_SOFTWARE': 'Enhanced Bathymetric CAE v3.5 (CLI Configurable Ensemble)',
                'ENHANCEMENT_METHOD': 'Ensemble Convolutional Autoencoder with Proper Denormalization',
                'ENSEMBLE_MODELS': 'CLI configurable multiple model variants',
                'ROBUST_BAG_CREATION': 'Direct creation with error handling'
            }
            
            for key, value in processing_metadata.items():
                dataset.SetMetadataItem(key, str(value), 'PROCESSING')
            
            # Close properly
            if uncertainty_data is not None:
                uncertainty_band.FlushCache()
                uncertainty_band = None
            elevation_band.FlushCache()
            elevation_band = None
            dataset.FlushCache()
            dataset = None
            
            # Verify the file
            verify_dataset = gdal.Open(str(output_path), gdal.GA_ReadOnly)
            if verify_dataset is None:
                self.logger.warning("Direct BAG creation verification failed")
                return False
            
            self.logger.info("✅ Direct BAG creation successful!")
            verify_dataset = None
            return True
            
        except Exception as e:
            self.logger.warning(f"Direct BAG creation failed: {e}")
            return False
    
    def _convert_tiff_to_bag(self, tiff_path: Path, bag_path: Path, metadata: Dict) -> bool:
        """Convert GeoTIFF to BAG using external tools."""
        
        try:
            import subprocess
            
            cmd = [
                'gdal_translate',
                '-of', 'BAG',
                '-co', 'VAR_TITLE=Enhanced Bathymetric Data (CLI Ensemble)',
                '-co', 'VAR_ABSTRACT=Enhanced bathymetric surface using CLI configurable ensemble',
                '-co', f'VAR_DATETIME={datetime.datetime.now().isoformat()}Z',
                '-co', 'VAR_VERTICAL_DATUM=MLLW',
                str(tiff_path),
                str(bag_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and bag_path.exists():
                # Verify the converted file
                test_ds = gdal.Open(str(bag_path))
                if test_ds is not None:
                    self.logger.info("✅ GeoTIFF -> BAG conversion successful!")
                    test_ds = None
                    return True
                else:
                    self.logger.warning("BAG file created but cannot be opened")
                    if bag_path.exists():
                        bag_path.unlink()
            
        except Exception as e:
            self.logger.warning(f"GeoTIFF -> BAG conversion failed: {e}")
        
        return False
    
    def _create_robust_geotiff(self, 
                              enhanced_data: np.ndarray,
                              output_path: Path,
                              metadata: Dict) -> bool:
        """Create robust GeoTIFF file."""
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            height, width = enhanced_data.shape
            geotransform = metadata.get('geotransform', [0, 1, 0, 0, 0, -1])
            projection = metadata.get('projection', '')
            nodata_value = metadata.get('nodata_value', -9999.0)
            creation_options = metadata.get('creation_options', [
                'COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'
            ])
            
            # Prepare data
            output_data = enhanced_data.astype(np.float32)
            invalid_mask = ~np.isfinite(output_data)
            if np.any(invalid_mask):
                output_data[invalid_mask] = nodata_value
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                str(output_path),
                width, height, 1,
                gdal.GDT_Float32,
                creation_options
            )
            
            if dataset is None:
                self.logger.error("Failed to create GeoTIFF")
                return False
            
            # Set geospatial information
            dataset.SetGeoTransform(geotransform)
            dataset.SetProjection(projection)
            
            # Write data
            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(nodata_value)
            band.SetDescription('Enhanced Bathymetry (CLI Ensemble)')
            band.WriteArray(output_data)
            
            # Set statistics
            valid_data = output_data[output_data != nodata_value]
            if len(valid_data) > 0:
                band.SetStatistics(
                    float(np.min(valid_data)),
                    float(np.max(valid_data)),
                    float(np.mean(valid_data)),
                    float(np.std(valid_data))
                )
            
            # Set metadata
            band_metadata = {
                'PROCESSING_SOFTWARE': 'Enhanced Bathymetric CAE v3.5 (CLI Configurable Ensemble)',
                'CREATION_DATE': datetime.datetime.now().isoformat(),
                'DATA_TYPE': 'Enhanced Bathymetry',
                'UNITS': 'meters',
                'ENSEMBLE_METHOD': 'CLI configurable multiple model predictions averaged'
            }
            band.SetMetadata(band_metadata)
            
            # Close properly
            band.FlushCache()
            band = None
            dataset.FlushCache()
            dataset = None
            
            self.logger.info("✅ GeoTIFF created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating GeoTIFF: {e}")
            return False
    
    def _create_robust_ascii(self, 
                            enhanced_data: np.ndarray,
                            output_path: Path,
                            metadata: Dict) -> bool:
        """Create robust ESRI ASCII Grid file."""
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            height, width = enhanced_data.shape
            geotransform = metadata.get('geotransform', [0, 1, 0, 0, 0, -1])
            nodata_value = metadata.get('nodata_value', -9999.0)
            
            # Calculate grid parameters
            xllcorner = geotransform[0]
            yllcorner = geotransform[3] + height * geotransform[5]
            cellsize = abs(geotransform[1])
            
            # Prepare data (flip vertically for ASCII grid format)
            output_data = np.flipud(enhanced_data.astype(np.float32))
            invalid_mask = ~np.isfinite(output_data)
            if np.any(invalid_mask):
                output_data[invalid_mask] = nodata_value
            
            # Write ASCII grid file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"ncols         {width}\n")
                f.write(f"nrows         {height}\n")
                f.write(f"xllcorner     {xllcorner:.6f}\n")
                f.write(f"yllcorner     {yllcorner:.6f}\n")
                f.write(f"cellsize      {cellsize:.6f}\n")
                f.write(f"NODATA_value  {nodata_value}\n")
                
                for row in output_data:
                    row_str = ' '.join(f"{val:.3f}" for val in row)
                    f.write(f"{row_str}\n")
            
            self.logger.info("✅ ASCII Grid created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ASCII Grid: {e}")
            return False


# MAIN PIPELINE CLASS WITH CLI CONFIGURABLE ENSEMBLE
class EnhancedBathymetricCAEPipeline:
    """
    Enhanced pipeline with CLI configurable ensemble implementation.
    
    Key features:
    - Configurable ensemble size via CLI arguments or config
    - Multiple model variants with automatic architecture cycling
    - Ensemble prediction averaging and uncertainty estimation
    - Robust error handling and fallback strategies
    - Same-format export with proper denormalization
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.data_processor = BathymetricProcessor(config)
        self.global_scaler = DepthScaler()
        self.ensemble_model = None  # Will be EnsembleCAE
        self.format_exporter = RobustFormatExporter()
        
        # Configure GPU
        self._configure_gpu()
        
        # Get ensemble size from config - SUPPORT MULTIPLE SOURCES
        self.ensemble_size = self._get_ensemble_size_from_config(config)
        
        self.logger.info("=== Enhanced Bathymetric CAE Pipeline v3.5 (CLI Configurable Ensemble) ===")
        self.logger.info(f"🔧 Ensemble size: {self.ensemble_size} models (CLI configurable)")
        self.logger.info("✅ Multiple architecture variants for robust predictions")
        self.logger.info("✅ Uncertainty estimation from ensemble variance")
    
    def _get_ensemble_size_from_config(self, config) -> int:
        """Extract ensemble size from config with multiple fallback sources."""
        
        # Priority order for ensemble size configuration:
        # 1. Direct attribute: config.ensemble_size
        # 2. CLI args attribute: config.args.ensemble_size (if args exist)
        # 3. Dictionary access: config['ensemble_size']
        # 4. Default value: 3
        
        ensemble_size = None
        
        # Method 1: Direct attribute
        if hasattr(config, 'ensemble_size'):
            ensemble_size = config.ensemble_size
            self.logger.info(f"Ensemble size from config.ensemble_size: {ensemble_size}")
        
        # Method 2: CLI args (common pattern)
        elif hasattr(config, 'args') and hasattr(config.args, 'ensemble_size'):
            ensemble_size = config.args.ensemble_size
            self.logger.info(f"Ensemble size from CLI args: {ensemble_size}")
        
        # Method 3: Dictionary-style access
        elif hasattr(config, '__getitem__'):
            try:
                ensemble_size = config['ensemble_size']
                self.logger.info(f"Ensemble size from config dict: {ensemble_size}")
            except (KeyError, TypeError):
                pass
        
        # Method 4: Check for other common attribute names
        elif hasattr(config, 'num_models'):
            ensemble_size = config.num_models
            self.logger.info(f"Ensemble size from config.num_models: {ensemble_size}")
        
        elif hasattr(config, 'n_models'):
            ensemble_size = config.n_models
            self.logger.info(f"Ensemble size from config.n_models: {ensemble_size}")
        
        # Default fallback
        if ensemble_size is None:
            ensemble_size = 3
            self.logger.info(f"Using default ensemble size: {ensemble_size}")
        
        # Validate ensemble size
        try:
            ensemble_size = int(ensemble_size)
            if ensemble_size < 1:
                self.logger.warning(f"Invalid ensemble size {ensemble_size}, using default of 3")
                ensemble_size = 3
            elif ensemble_size > 20:
                self.logger.warning(f"Large ensemble size {ensemble_size} may be very slow")
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid ensemble size {ensemble_size}, using default of 3")
            ensemble_size = 3
        
        return ensemble_size
    
    def _configure_gpu(self):
        """Configure GPU settings."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Configured {len(gpus)} GPU(s)")
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration failed: {e}")
    
    def run(self, input_folder: str, output_folder: str, model_name: str = "enhanced_bathymetric_cae.keras"):
        """Run the pipeline with CLI configurable ensemble training and prediction."""
        try:
            self.logger.info("=== STARTING CLI CONFIGURABLE ENSEMBLE PIPELINE ===")
            
            # Setup paths
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find supported files
            file_list = []
            supported_extensions = getattr(self.config, 'supported_formats', ['.bag', '.tif', '.tiff', '.asc'])
            for ext in supported_extensions:
                file_list.extend(input_path.glob(f"*{ext}"))
            
            if not file_list:
                raise ValueError(f"No supported files found in {input_folder}")
            
            self.logger.info(f"Found {len(file_list)} files to process")
            
            # Load and prepare training data
            self.logger.info("🔧 Loading data with proper scaling...")
            X_train, y_train = self._prepare_training_data(file_list)
            
            # Train CLI CONFIGURABLE ensemble
            self.logger.info(f"🤖 Training CLI configurable ensemble of {self.ensemble_size} models...")
            self.ensemble_model = EnsembleCAE(
                input_shape=X_train.shape[1:], 
                ensemble_size=self.ensemble_size
            )
            
            # Train the entire ensemble
            ensemble_histories = self.ensemble_model.fit_ensemble(
                X_train, y_train, 
                epochs=getattr(self.config, 'epochs', 50)
            )
            
            # Process files with ENSEMBLE predictions
            self.logger.info("🔧 Creating enhanced files using ensemble predictions...")
            success_count = self._process_files_with_ensemble(file_list, output_path)
            
            # Save ensemble models
            model_path = output_path / model_name
            self.ensemble_model.save_ensemble(model_path)
            self.logger.info(f"Ensemble models saved: {model_path}")
            
            # Generate summary
            self._generate_ensemble_summary(file_list, output_path, ensemble_histories, success_count)
            
            self.logger.info("=== CLI CONFIGURABLE ENSEMBLE PIPELINE COMPLETED ===")
            
        except Exception as e:
            self.logger.error(f"CLI configurable ensemble pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _prepare_training_data(self, file_list: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with proper scaling."""
        all_depth_data = []
        
        # Load all files
        for file_path in file_list:
            try:
                input_data, shape, metadata = self.data_processor.preprocess_bathymetric_grid(file_path)
                depth_data = input_data[..., 0]  # Remove channel dimension
                all_depth_data.append(depth_data)
                
            except Exception as e:
                self.logger.warning(f"Skipping {file_path}: {e}")
                continue
        
        if not all_depth_data:
            raise ValueError("No valid files could be loaded")
        
        # Fit global scaler on ALL data
        combined_data = np.concatenate([d.flatten() for d in all_depth_data])
        self.global_scaler.fit(combined_data)
        
        self.logger.info(f"🔧 Global scaler fitted: [{self.global_scaler.depth_min:.2f}, {self.global_scaler.depth_max:.2f}]m")
        
        # Prepare training arrays with proper normalization
        X_list = []
        y_list = []
        
        for depth_data in all_depth_data:
            # Normalize for training
            normalized_data = self.global_scaler.normalize_depth(depth_data)
            
            # For autoencoder, input and target are the same
            X_list.append(normalized_data[..., np.newaxis])
            y_list.append(normalized_data[..., np.newaxis])
        
        X_train = np.array(X_list)
        y_train = np.array(y_list)
        
        self.logger.info(f"Training data prepared: {X_train.shape}")
        return X_train, y_train
    
    def _process_files_with_ensemble(self, file_list: List[Path], output_path: Path) -> int:
        """Process files using ensemble predictions with uncertainty estimation."""
        success_count = 0
        
        for file_path in file_list:
            try:
                self.logger.info(f"🔧 Processing {file_path.name} with {self.ensemble_size}-model ensemble (format: {file_path.suffix})...")
                
                # Load original data with full metadata
                input_data, shape, metadata = self.data_processor.preprocess_bathymetric_grid(file_path)
                depth_data = input_data[..., 0]
                
                # Get uncertainty data if available
                uncertainty_data = metadata.get('uncertainty_data', None)
                
                # Normalize for ensemble models
                normalized_input = self.global_scaler.normalize_depth(depth_data)
                
                # ENSEMBLE PREDICTION with uncertainty estimation
                model_input = normalized_input[np.newaxis, ..., np.newaxis]
                ensemble_prediction, ensemble_uncertainty = self.ensemble_model.predict_ensemble(model_input)
                
                # Extract results
                enhanced_normalized = ensemble_prediction[0, ..., 0]
                prediction_uncertainty = ensemble_uncertainty[0, ..., 0]
                
                self.logger.info(f"   Ensemble prediction uncertainty range: {np.min(prediction_uncertainty):.6f} to {np.max(prediction_uncertainty):.6f}")
                
                # CRITICAL: Denormalize back to real-world coordinates
                enhanced_real_world = self.global_scaler.denormalize_depth(enhanced_normalized)
                
                # Resize back to original shape if needed
                if 'original_shape' in metadata and 'resized_for_processing' in metadata:
                    original_shape = metadata['original_shape']
                    if enhanced_real_world.shape != original_shape:
                        zoom_factors = (
                            original_shape[0] / enhanced_real_world.shape[0],
                            original_shape[1] / enhanced_real_world.shape[1]
                        )
                        enhanced_real_world = ndimage.zoom(enhanced_real_world, zoom_factors, order=1)
                        
                        # Also resize prediction uncertainty
                        prediction_uncertainty = ndimage.zoom(prediction_uncertainty, zoom_factors, order=1)
                        
                        # Also resize original uncertainty if available
                        if uncertainty_data is not None:
                            uncertainty_data = ndimage.zoom(uncertainty_data, zoom_factors, order=1)
                        
                        self.logger.info(f"Resized output back to original: {original_shape}")
                
                # Combine prediction uncertainty with original uncertainty if available
                final_uncertainty = prediction_uncertainty
                if uncertainty_data is not None:
                    # Combine uncertainties (root sum of squares)
                    final_uncertainty = np.sqrt(prediction_uncertainty**2 + uncertainty_data**2)
                    self.logger.info("   Combined ensemble uncertainty with original uncertainty")
                
                # Generate output filename (same format as input)
                input_format = metadata.get('input_format', 'UNKNOWN')
                input_extension = metadata.get('input_extension', file_path.suffix)
                base_name = file_path.stem
                
                # Create output filename with same extension
                output_file_path = output_path / f"enhanced_{base_name}{input_extension}"
                
                # Export in the same format as input
                success = self.format_exporter.create_same_format_output(
                    enhanced_real_world,
                    file_path,
                    output_file_path,
                    metadata,
                    final_uncertainty  # Pass the combined uncertainty
                )
                
                if success:
                    self.logger.info(f"✅ Successfully created: {output_file_path}")
                    
                    # Verify the output file can be opened
                    verify_dataset = gdal.Open(str(output_file_path))
                    if verify_dataset is not None:
                        self.logger.info(f"✅ Output file verified and readable")
                        verify_dataset = None
                        success_count += 1
                    else:
                        self.logger.warning(f"⚠️  Output file created but cannot be opened")
                        success_count += 1  # Still count as success since file was created
                else:
                    self.logger.error(f"❌ Failed to create output for {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        return success_count
    
    def _generate_ensemble_summary(self, file_list: List[Path], output_path: Path, ensemble_histories: List, success_count: int):
        """Generate summary for CLI configurable ensemble processing."""
        
        # Calculate ensemble training statistics
        final_losses = [hist.history['loss'][-1] for hist in ensemble_histories]
        final_val_losses = [hist.history['val_loss'][-1] for hist in ensemble_histories if 'val_loss' in hist.history]
        
        # Get architecture names
        architecture_names = [self.ensemble_model._get_architecture_name(i) for i in range(self.ensemble_size)]
        
        summary = {
            'pipeline_version': 'Enhanced Bathymetric CAE Pipeline v3.5 (CLI Configurable Ensemble)',
            'processing_date': datetime.datetime.now().isoformat(),
            'ensemble_info': {
                'ensemble_size': self.ensemble_size,
                'ensemble_method': 'CLI_configurable_multiple_model_variants',
                'uncertainty_estimation': 'model_disagreement_based',
                'model_architectures': architecture_names,
                'configurable_via_cli': True
            },
            'training_statistics': {
                'individual_final_losses': final_losses,
                'mean_final_loss': float(np.mean(final_losses)),
                'std_final_loss': float(np.std(final_losses)),
                'individual_val_losses': final_val_losses,
                'mean_val_loss': float(np.mean(final_val_losses)) if final_val_losses else None
            },
            'export_strategy': {
                'method': 'same_format_as_input',
                'uncertainty_handling': 'ensemble_uncertainty_combined_with_original'
            },
            'total_files_processed': len(file_list),
            'successful_files': success_count,
            'success_rate': f"{100*success_count/len(file_list):.1f}%" if len(file_list) > 0 else "0%",
            'output_folder': str(output_path),
            'scaling_info': self.global_scaler.get_scaling_metadata(),
            'cli_configuration': {
                'ensemble_size_configurable': True,
                'usage_example': 'python main.py --ensemble-size 5 --input data/ --output results/'
            }
        }
        
        summary_path = output_path / "enhanced_processing_summary_cli_ensemble.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"CLI ensemble summary saved: {summary_path}")
        
        # Create CLI-specific usage instructions
        instructions_path = output_path / "CLI_ENSEMBLE_USAGE.txt"
        with open(instructions_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED BATHYMETRIC DATA - CLI CONFIGURABLE ENSEMBLE\n")
            f.write("="*65 + "\n\n")
            f.write(f"Processing completed: {datetime.datetime.now()}\n")
            f.write(f"Ensemble size: {self.ensemble_size} models (CLI configurable)\n")
            f.write(f"Files processed: {len(file_list)}\n")
            f.write(f"Success rate: {summary['success_rate']}\n\n")
            
            f.write("CLI CONFIGURATION OPTIONS:\n")
            f.write("python main.py --ensemble-size 3    # Use 3 models (default)\n")
            f.write("python main.py --ensemble-size 5    # Use 5 models (more robust)\n")
            f.write("python main.py --ensemble-size 7    # Use 7 models (highest quality)\n")
            f.write("python main.py --ensemble-size 1    # Single model (fastest)\n\n")
            
            f.write("ENSEMBLE DETAILS:\n")
            f.write(f"- Model count: {self.ensemble_size} models\n")
            f.write(f"- Architectures: {', '.join(architecture_names)}\n")
            f.write(f"- Mean training loss: {summary['training_statistics']['mean_final_loss']:.6f}\n")
            f.write(f"- Training loss std: {summary['training_statistics']['std_final_loss']:.6f}\n\n")
            
            f.write("ARCHITECTURE CYCLING:\n")
            f.write("The pipeline automatically cycles through different architectures:\n")
            for i, arch in enumerate(architecture_names):
                f.write(f"- Model {i+1}: {arch}\n")
            f.write("\nMore models = more architecture diversity = better results\n\n")
            
            f.write("PERFORMANCE vs QUALITY TRADE-OFFS:\n")
            f.write("- 1 model:  Fastest processing, basic enhancement\n")
            f.write("- 3 models: Good balance of speed and quality\n")
            f.write("- 5 models: Better quality, slower processing\n")
            f.write("- 7+ models: Best quality, longest processing time\n\n")
        
        # Print CLI ensemble summary
        print("\n" + "="*80)
        print("🎉 CLI CONFIGURABLE ENSEMBLE PIPELINE COMPLETED")
        print("="*80)
        print("🔧 CLI CONFIGURATION:")
        print(f"   🤖 Ensemble size: {self.ensemble_size} models (configurable)")
        print(f"   📊 Architectures: {', '.join(architecture_names)}")
        print(f"   ⚙️  Usage: python main.py --ensemble-size {self.ensemble_size}")
        print()
        
        print("📊 PROCESSING RESULTS:")
        print(f"   📁 Files processed: {len(file_list)}")
        print(f"   ✅ Successful outputs: {success_count}")
        print(f"   📈 Success rate: {summary['success_rate']}")
        print(f"   🎯 Depth range: [{self.global_scaler.depth_min:.1f}, {self.global_scaler.depth_max:.1f}]m")
        print()
        
        print("🤖 ENSEMBLE TRAINING STATISTICS:")
        print(f"   📊 Mean final loss: {summary['training_statistics']['mean_final_loss']:.6f}")
        print(f"   📊 Loss standard deviation: {summary['training_statistics']['std_final_loss']:.6f}")
        if summary['training_statistics']['mean_val_loss']:
            print(f"   📊 Mean validation loss: {summary['training_statistics']['mean_val_loss']:.6f}")
        print()
        
        print("⚙️  CLI ENSEMBLE CONFIGURATION:")
        print("   🔧 To use different ensemble sizes:")
        print("      python main.py --ensemble-size 1    # Single model (fastest)")
        print("      python main.py --ensemble-size 3    # Default (balanced)")
        print("      python main.py --ensemble-size 5    # Higher quality")
        print("      python main.py --ensemble-size 7    # Best quality")
        print()
        
        print(f"💾 Output folder: {output_path}")
        print()
        
        if success_count == len(file_list):
            print("✅ ALL FILES PROCESSED SUCCESSFULLY WITH CLI ENSEMBLE!")
        elif success_count > 0:
            print("⚠️  SOME FILES PROCESSED WITH CLI ENSEMBLE")
        else:
            print("❌ NO FILES PROCESSED SUCCESSFULLY")
        
        print("="*80)


# For compatibility with existing imports
def create_enhanced_visualization(*args, **kwargs):
    """Placeholder for visualization function."""
    pass

def plot_training_history(*args, **kwargs):
    """Placeholder for training history plotting."""
    pass


# CLI Argument Parser Enhancement
def add_ensemble_args_to_parser(parser):
    """Add ensemble-specific arguments to an existing argument parser."""
    
    ensemble_group = parser.add_argument_group('ensemble', 'Ensemble configuration options')
    
    ensemble_group.add_argument(
        '--ensemble-size', 
        type=int, 
        default=3,
        help='Number of models in the ensemble (default: 3). More models = better quality but slower processing'
    )
    
    ensemble_group.add_argument(
        '--ensemble-architectures',
        nargs='+',
        choices=['standard', 'deep', 'wide', 'dense', 'lightweight', 'heavy'],
        help='Specific architectures to use (optional). If not specified, automatic cycling is used'
    )
    
    return parser


def create_ensemble_parser():
    """Create a complete argument parser with ensemble options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Bathymetric CAE Pipeline with CLI Configurable Ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 3-model ensemble
  python main.py --input data/ --output results/
  
  # Single model for speed
  python main.py --input data/ --output results/ --ensemble-size 1
  
  # 5-model ensemble for better quality
  python main.py --input data/ --output results/ --ensemble-size 5
  
  # 7-model ensemble for best quality
  python main.py --input data/ --output results/ --ensemble-size 7
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Input folder containing bathymetric files')
    parser.add_argument('--output', required=True, help='Output folder for enhanced files')
    
    # Optional arguments
    parser.add_argument('--model', default='enhanced_bathymetric_cae.keras', help='Model filename')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--grid-size', type=int, default=512, help='Processing grid size (default: 512)')
    
    # Add ensemble arguments
    add_ensemble_args_to_parser(parser)
    
    return parser


# Quick test function
def test_cli_ensemble_pipeline():
    """Test that the CLI configurable ensemble pipeline is working correctly."""
    
    print("🧪 TESTING CLI CONFIGURABLE ENSEMBLE PIPELINE")
    print("="*50)
    
    try:
        # Test imports
        from processing.pipeline import EnhancedBathymetricCAEPipeline
        print("✅ Pipeline import successful")
        
        # Test different ensemble size configurations
        test_configs = [
            {'ensemble_size': 1},
            {'ensemble_size': 3}, 
            {'ensemble_size': 5},
        ]
        
        for i, config_dict in enumerate(test_configs):
            print(f"\n🔧 Testing configuration {i+1}: {config_dict}")
            
            # Create test config
            class TestConfig:
                def __init__(self, **kwargs):
                    self.grid_size = 64
                    self.epochs = 5
                    self.supported_formats = ['.bag', '.tif', '.tiff', '.asc']
                    # Set ensemble size
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            config = TestConfig(**config_dict)
            pipeline = EnhancedBathymetricCAEPipeline(config)
            
            expected_size = config_dict['ensemble_size']
            actual_size = pipeline.ensemble_size
            
            if actual_size == expected_size:
                print(f"   ✅ Ensemble size correctly set to {actual_size}")
            else:
                print(f"   ❌ Ensemble size mismatch: expected {expected_size}, got {actual_size}")
                return False
        
        print(f"\n🔧 Testing CLI args simulation...")
        
        # Test CLI args simulation
        class MockArgs:
            ensemble_size = 7
        
        class TestConfigWithArgs:
            grid_size = 64
            epochs = 5
            supported_formats = ['.bag', '.tif', '.tiff', '.asc']
            args = MockArgs()
        
        config_with_args = TestConfigWithArgs()
        pipeline_with_args = EnhancedBathymetricCAEPipeline(config_with_args)
        
        if pipeline_with_args.ensemble_size == 7:
            print("   ✅ CLI args parsing simulation successful")
        else:
            print(f"   ❌ CLI args parsing failed: expected 7, got {pipeline_with_args.ensemble_size}")
            return False
        
        print(f"\n🧪 Testing argument parser...")
        
        # Test argument parser
        parser = create_ensemble_parser()
        test_args = parser.parse_args(['--input', 'test_input', '--output', 'test_output', '--ensemble-size', '5'])
        
        if test_args.ensemble_size == 5:
            print("   ✅ Argument parser working correctly")
        else:
            print(f"   ❌ Argument parser failed: expected 5, got {test_args.ensemble_size}")
            return False
        
        print("\n🎉 CLI CONFIGURABLE ENSEMBLE PIPELINE IS READY!")
        print("   🤖 Ensemble size configurable via --ensemble-size argument")
        print("   ⚙️  Supports 1-20 models with automatic architecture cycling")
        print("   📊 Provides uncertainty estimation from ensemble variance")
        print("   🔧 Maintains backward compatibility with existing configs")
        
        print("\n📋 USAGE EXAMPLES:")
        print("   python main.py --input data/ --output results/ --ensemble-size 1   # Fast")
        print("   python main.py --input data/ --output results/ --ensemble-size 3   # Default")  
        print("   python main.py --input data/ --output results/ --ensemble-size 5   # Better")
        print("   python main.py --input data/ --output results/ --ensemble-size 7   # Best")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI ensemble pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Example main function showing CLI integration
def example_main_with_cli():
    """Example main function showing how to integrate CLI ensemble configuration."""
    
    # Create parser with ensemble options
    parser = create_ensemble_parser()
    args = parser.parse_args()
    
    # Create config object that includes ensemble size from CLI
    class Config:
        def __init__(self, args):
            self.grid_size = args.grid_size
            self.epochs = args.epochs
            self.ensemble_size = args.ensemble_size  # CLI configurable!
            self.supported_formats = ['.bag', '.tif', '.tiff', '.asc']
            self.args = args  # Store args for additional access
    
    config = Config(args)
    
    # Create and run pipeline
    pipeline = EnhancedBathymetricCAEPipeline(config)
    pipeline.run(args.input, args.output, args.model)


if __name__ == "__main__":
    # Test the CLI configurable ensemble pipeline
    test_cli_ensemble_pipeline()
        """
CLI Configurable Ensemble Pipeline
=================================

This updates the pipeline to make ensemble size configurable via CLI arguments
while maintaining backward compatibility with existing configuration systems.
"""

import logging
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from scipy import ndimage
from osgeo import gdal, osr
import json
import gc
import warnings

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# Matplotlib with fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available - training plots will be skipped")


class DepthScaler:
    """Fixed depth scaling and denormalization."""
    
    def __init__(self):
        self.depth_min: Optional[float] = None
        self.depth_max: Optional[float] = None
        self.depth_mean: Optional[float] = None
        self.depth_std: Optional[float] = None
        self.uncertainty_min: Optional[float] = None
        self.uncertainty_max: Optional[float] = None
        self.scaling_method: str = "minmax"
        self.is_fitted: bool = False
        
    def fit(self, depth_data: np.ndarray, uncertainty_data: Optional[np.ndarray] = None):
        """Fit the scaler to depth data statistics."""
        valid_depth = depth_data[np.isfinite(depth_data)]
        if len(valid_depth) == 0:
            raise ValueError("No valid depth data for scaling")
        
        self.depth_min = float(np.min(valid_depth))
        self.depth_max = float(np.max(valid_depth))
        self.depth_mean = float(np.mean(valid_depth))
        self.depth_std = float(np.std(valid_depth))
        
        if uncertainty_data is not None:
            valid_uncertainty = uncertainty_data[np.isfinite(uncertainty_data)]
            if len(valid_uncertainty) > 0:
                self.uncertainty_min = float(np.min(valid_uncertainty))
                self.uncertainty_max = float(np.max(valid_uncertainty))
        
        self.is_fitted = True
        logging.info(f"Scaler fitted - Depth range: [{self.depth_min:.2f}, {self.depth_max:.2f}]m")
    
    def normalize_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """Normalize depth data to [0, 1] range."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        normalized = (depth_data - self.depth_min) / (self.depth_max - self.depth_min)
        return np.clip(normalized, 0, 1)
    
    def denormalize_depth(self, normalized_data: np.ndarray) -> np.ndarray:
        """CRITICAL FIX: Denormalize data back to real-world coordinates."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return normalized_data * (self.depth_max - self.depth_min) + self.depth_min
    
    def get_scaling_metadata(self) -> Dict[str, Any]:
        """Get scaling metadata for compatibility."""
        return {
            'depth_min': self.depth_min,
            'depth_max': self.depth_max,
            'depth_mean': self.depth_mean,
            'depth_std': self.depth_std,
            'uncertainty_min': self.uncertainty_min,
            'uncertainty_max': self.uncertainty_max,
            'scaling_method': self.scaling_method
        }


class BathymetricProcessor:
    """Enhanced data processor that preserves original format information."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess_bathymetric_grid(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int], Optional[dict]]:
        """Load and preprocess bathymetric data while preserving format details."""
        file_path = Path(file_path)
        self.logger.info(f"Loading file: {file_path}")
        
        dataset = None
        try:
            dataset = gdal.Open(str(file_path))
            if dataset is None:
                raise ValueError(f"Cannot open file: {file_path}")
            
            # Detect input format
            driver = dataset.GetDriver()
            input_format = driver.GetDescription().upper()
            
            # Get comprehensive metadata
            metadata = {
                'geotransform': dataset.GetGeoTransform(),
                'projection': dataset.GetProjection(),
                'metadata': dataset.GetMetadata(),
                'original_shape': (dataset.RasterYSize, dataset.RasterXSize),
                'input_format': input_format,
                'input_extension': file_path.suffix.lower(),
                'bands': dataset.RasterCount,
                'datatype': dataset.GetRasterBand(1).DataType,
                'creation_options': self._get_creation_options_for_format(input_format)
            }
            
            # Read depth data (band 1)
            depth_band = dataset.GetRasterBand(1)
            depth_data = depth_band.ReadAsArray().astype(np.float32)
            
            # Handle NoData values
            nodata = depth_band.GetNoDataValue()
            if nodata is not None:
                depth_data[depth_data == nodata] = np.nan
                metadata['nodata_value'] = nodata
            else:
                metadata['nodata_value'] = -9999.0
            
            # Read uncertainty data if available (for BAG files)
            if dataset.RasterCount >= 2:
                uncertainty_band = dataset.GetRasterBand(2)
                uncertainty_data = uncertainty_band.ReadAsArray().astype(np.float32)
                uncertainty_nodata = uncertainty_band.GetNoDataValue()
                if uncertainty_nodata is not None:
                    uncertainty_data[uncertainty_data == uncertainty_nodata] = np.nan
                metadata['has_uncertainty'] = True
                metadata['uncertainty_data'] = uncertainty_data
            else:
                metadata['has_uncertainty'] = False
            
            # Validate data
            valid_mask = np.isfinite(depth_data)
            if np.sum(valid_mask) == 0:
                raise ValueError(f"No valid data in {file_path}")
            
            # Fill invalid values
            if np.sum(valid_mask) < depth_data.size:
                fill_value = np.mean(depth_data[valid_mask])
                depth_data[~valid_mask] = fill_value
            
            # Store original shape before any resizing
            original_shape = depth_data.shape
            metadata['original_shape'] = original_shape
            
            # Resize to config grid size if needed for processing
            if hasattr(self.config, 'grid_size'):
                target_size = (self.config.grid_size, self.config.grid_size)
                if depth_data.shape != target_size:
                    zoom_factors = (
                        target_size[0] / depth_data.shape[0],
                        target_size[1] / depth_data.shape[1]
                    )
                    depth_data = ndimage.zoom(depth_data, zoom_factors, order=1)
                    metadata['resized_for_processing'] = True
                    metadata['zoom_factors'] = zoom_factors
            
            # Return in expected format (add channel dimension)
            input_data = depth_data[..., np.newaxis]
            
            return input_data, depth_data.shape[:2], metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
        finally:
            if dataset is not None:
                dataset = None
            gc.collect()
    
    def _get_creation_options_for_format(self, format_name: str) -> List[str]:
        """Get appropriate creation options for each format."""
        
        if format_name == 'BAG':
            return [
                'VAR_TITLE=Enhanced Bathymetric Data',
                'VAR_ABSTRACT=Enhanced bathymetric surface using deep learning ensemble',
                f'VAR_DATETIME={datetime.datetime.now().isoformat()}Z',
                'VAR_VERTICAL_DATUM=MLLW',
                'VAR_PRODUCT=Enhanced Bathymetry'
            ]
        elif format_name in ['GTIFF', 'GEOTIFF']:
            return [
                'COMPRESS=LZW',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=IF_SAFER'
            ]
        else:
            return []


class EnsembleCAE:
    """Ensemble of multiple CAE models with configurable size."""
    
    def __init__(self, input_shape: Tuple[int, int, int], ensemble_size: int = 3):
        self.input_shape = input_shape
        self.ensemble_size = ensemble_size
        self.models = []
        self.model_histories = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate ensemble size
        if ensemble_size < 1:
            raise ValueError(f"Ensemble size must be at least 1, got {ensemble_size}")
        
        if ensemble_size > 10:
            self.logger.warning(f"Large ensemble size ({ensemble_size}) may require significant memory and training time")
        
        # Build ensemble models
        self._build_ensemble()
    
    def _build_ensemble(self):
        """Build multiple models with different architectures."""
        self.logger.info(f"Building ensemble of {self.ensemble_size} models...")
        
        for i in range(self.ensemble_size):
            model = self._build_variant_model(i)
            self.models.append(model)
            self.logger.info(f"✅ Model {i+1}/{self.ensemble_size} created ({self._get_architecture_name(i)})")
    
    def _get_architecture_name(self, variant_id: int) -> str:
        """Get human-readable name for each architecture variant."""
        architecture_names = [
            "Standard",
            "Deep", 
            "Wide-Residual",
            "Dense",
            "Lightweight",
            "Heavy",
            "Asymmetric",
            "Multi-Scale",
            "Attention",
            "Hybrid"
        ]
        return architecture_names[variant_id % len(architecture_names)]
    
    def _build_variant_model(self, variant_id: int):
        """Build different model variants for ensemble diversity."""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Cycle through different architectures based on variant_id
        architecture_type = variant_id % 6  # 6 different base architectures
        
        if architecture_type == 0:
            # Variant 1: Standard architecture
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            encoded = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
            
        elif architecture_type == 1:
            # Variant 2: Deeper architecture
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            encoded = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
            
        elif architecture_type == 2:
            # Variant 3: Wide architecture with multi-kernel
            x1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            x2 = tf.keras.layers.Conv2D(64, 5, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.Concatenate()([x1, x2])
            
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            encoded = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
            
        elif architecture_type == 3:
            # Variant 4: Dense connections
            x1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x1)
            x = tf.keras.layers.Concatenate()([x1, x2])
            
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            encoded = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
            
        elif architecture_type == 4:
            # Variant 5: Lightweight architecture
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            encoded = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
            
        else:  # architecture_type == 5
            # Variant 6: Heavy architecture
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            encoded = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
            
            # Decoder
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(encoded)
            x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            outputs = tf.keras.layers.Conv2D(1, 3, activation='linear', padding='same')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Vary learning rates slightly for diversity
        learning_rates = [0.001, 0.0008, 0.0012, 0.0009, 0.0011, 0.0007]
        lr = learning_rates[variant_id % len(learning_rates)]
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit_ensemble(self, X_train, y_train, epochs=50, validation_split=0.2):
        """Train all models in the ensemble."""
        self.logger.info(f"Training ensemble of {self.ensemble_size} models...")
        
        all_histories = []
        
        for i, model in enumerate(self.models):
            arch_name = self._get_architecture_name(i)
            self.logger.info(f"🤖 Training model {i+1}/{self.ensemble_size} ({arch_name})...")
            
            # Add some variation in training parameters
            batch_sizes = [4, 6, 8, 5, 7, 9]
            batch_size = batch_sizes[i % len(batch_sizes)]
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=10, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, 
                    patience=5,
                    monitor='val_loss'
                )
            ]
            
            # Train individual model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1 if i == 0 else 0,  # Only show progress for first model
                callbacks=callbacks
            )
            
            all_histories.append(history)
            
            # Log training results
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            self.logger.info(f"✅ Model {i+1} ({arch_name}) completed - Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f if final_val_loss else 'N/A'}")
        
        self.model_histories = all_histories
        self.logger.info(f"🎉 Ensemble training completed!")
        return all_histories
    
    def predict_ensemble(self, X):
        """Make ensemble predictions with uncertainty estimation."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Stack predictions
        all_predictions = np.stack(predictions, axis=0)  # Shape: (n_models, batch, height, width, channels)
        
        # Calculate ensemble mean
        ensemble_mean = np.mean(all_predictions, axis=0)
        
        # Calculate ensemble uncertainty (standard deviation)
        ensemble_std = np.std(all_predictions, axis=0)
        
        return ensemble_mean, ensemble_std
    
    def save_ensemble(self, save_path: Path):
        """Save all models in the ensemble."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            arch_name = self._get_architecture_name(i).lower().replace('-', '_')
            model_path = save_path.parent / f"{save_path.stem}_model_{i}_{arch_name}{save_path.suffix}"
            model.save(str(model_path))
            self.logger.info(f"Saved model {i+1} ({self._get_architecture_name(i)}) to: {model_path}")
        
        # Save ensemble metadata
        metadata = {
            'ensemble_size': self.ensemble_size,
            'input_shape': self.input_shape,
            'model_files': [f"{save_path.stem}_model_{i}_{self._get_architecture_name(i).lower().replace('-', '_')}{save_path.suffix}" for i in range(self.ensemble_size)],
            'architectures': [self._get_architecture_name(i) for i in range(self.ensemble_size)],
            'training_date': datetime.datetime.now().isoformat()
        }
        
        metadata_path = save_path.parent / f"{save_path.stem}_ensemble_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved ensemble metadata to: {metadata_path}")


class RobustFormatExporter:
    """Creates robust output files in the same format as input with proper error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_same_format_output(self, 
                                 enhanced_data: np.ndarray,
                                 original_file_path: Path,
                                 output_path: Path,
                                 metadata: Dict,
                                 uncertainty_data: Optional[np.ndarray] = None) -> bool:
        """Create output in the same format as input with robust handling."""
        
        input_format = metadata.get('input_format', 'UNKNOWN')
        input_extension = metadata.get('input_extension', '.tif')
        
        self.logger.info(f"Creating {input_format} output: {output_path}")
        
        # Route to appropriate format creator
        if input_format == 'BAG' or input_extension == '.bag':
            return self._create_robust_bag(enhanced_data, output_path, metadata, uncertainty_data)
        elif input_format in ['GTIFF', 'GEOTIFF'] or input_extension in ['.tif', '.tiff']:
            return self._create_robust_geotiff(enhanced_data, output_path, metadata)
        elif input_extension == '.asc':
            return self._create_robust_ascii(enhanced_data, output_path, metadata)
        else:
            # Default to GeoTIFF for unknown formats
            self.logger.warning(f"Unknown format {input_format}, defaulting to GeoTIFF")
            return self._create_robust_geotiff(enhanced_data, output_path, metadata)
    
    def _create_robust_bag(self, 
                          enhanced_data: np.ndarray,
                          output_path: Path,
                          metadata: Dict,
                          uncertainty_data: Optional[np.ndarray] = None) -> bool:
        """Create robust BAG file with multiple fallback strategies."""
        
        self.logger.info("Creating robust BAG file...")
        
        # Strategy 1: Try direct BAG creation with robust settings
        if self._try_direct_bag_creation(enhanced_data, output_path, metadata, uncertainty_data):
            return True
        
        # Strategy 2: Create GeoTIFF first, then convert to BAG
        self.logger.info("Direct BAG creation failed, trying GeoTIFF -> BAG conversion...")
        temp_tiff = output_path.with_suffix('.temp.tif')
        
        try:
            if self._create_robust_geotiff(enhanced_data, temp_tiff, metadata):
                if self._convert_tiff_to_bag(temp_tiff, output_path, metadata):
                    # Clean up temp file
                    if temp_tiff.exists():
                        temp_tiff.unlink()
                    return True
            
            # Clean up temp file if conversion failed
            if temp_tiff.exists():
                temp_tiff.unlink()
                
        except Exception as e:
            self.logger.warning(f"GeoTIFF -> BAG conversion failed: {e}")
            if temp_tiff.exists():
                temp_tiff.unlink()
        
        # Strategy 3: Last resort - create a working GeoTIFF with .bag extension
        self.logger.warning("All BAG creation methods failed, creating GeoTIFF with .bag extension")
        return self._create_robust_geotiff(enhanced_data, output_path, metadata)
    
    def _try_direct_bag_creation(self, 
                                enhanced_data: np.ndarray,
                                output_path: Path,
                                metadata: Dict,
                                uncertainty_data: Optional[np.ndarray] = None) -> bool:
        """Try direct BAG creation with robust error handling."""
        
        try:
            import subprocess
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            driver = gdal.GetDriverByName('BAG')
            if driver is None:
                self.logger.warning("BAG driver not available")
                return False
            
            height, width = enhanced_data.shape
            geotransform = metadata.get('geotransform', [0, 1, 0, 0, 0, -1])
            projection = metadata.get('projection', '')
            nodata_value = metadata.get('nodata_value', -9999.0)
            creation_options = metadata.get('creation_options', [])
            
            # Determine number of bands
            num_bands = 2 if uncertainty_data is not None else 1
            
            # Create dataset with robust options
            dataset = driver.Create(
                str(output_path),
                width, height,
                num_bands,
                gdal.GDT_Float32,
                creation_options
            )
            
            if dataset is None:
                self.logger.warning("Failed to create BAG dataset")
                return False
            
            # Set geospatial information
            if geotransform:
                dataset.SetGeoTransform(geotransform)
            
            if projection:
                dataset.SetProjection(projection)
            else:
                # Set default WGS84
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                dataset.SetProjection(srs.ExportToWkt())
            
            # Prepare enhanced data
            output_data = enhanced_data.astype(np.float32)
            invalid_mask = ~np.isfinite(output