# SenseLess Deployment Pipeline

## Short system description
SenseLess is a hybrid anomaly detection framework where non-vision sensors run continuously as the primary detector, and the vision model activates only when sensor confidence is low, errors are detected, or drift is suspected. Vision outputs both validate decisions and provide trusted feedback for self-healing retraining of the non-vision model.

## Long description
The paper introduces *SenseLess*, a hybrid anomaly detection framework in which non-vision sensors operate continuously as the primary detection modality, while vision is used sparingly and conditionally to augment non-vision decisions. To enable this visual augmentation, the system must first train a reliable vision-based anomaly detector, which requires labelled image data that are typically unavailable in indoor environments. During training, SenseLess automatically generates image labels without manual annotation by combining sensor-guided anomaly detection and self-supervised visual clustering. These components are integrated through a confidence-weighted label refinement process that yields a high-quality self-labelled image dataset. This design addresses the scarcity of annotated image data highlighted in the thesis and establishes the visual models required to support selective, confidence-driven camera activation during deployment.

A central methodological contribution of the paper is the introduction of the Hierarchical Event-Driven Synchronization (HEDS) algorithm, which estimates and compensates for heterogeneous sensor response delays during sensor–image alignment. As demonstrated through both quantitative evaluation and ablation analysis, delay-aware alignment improves labeling accuracy in scenarios characterised by gradual environmental responses, such as door openings and occupancy changes. This contribution operationalises the observation, established through empirical analysis in this work, that temporal misalignment between sensor and image data is a critical barrier to effective multimodal learning and must be addressed to enable reliable cross-modal supervision.

During deployment, the system adopts a confidence-aware decision pipeline in which the non-vision anomaly detector runs continuously, and the vision model is activated only when sensor predictions are uncertain, erroneous, or affected by drift. Vision-based decisions are used both for immediate validation and as trusted feedback signals for retraining the non-vision model through a self-healing mechanism. Experimental results show that this selective activation strategy limits visual processing to less than 4% of operating time while enabling recovery from performance degradation caused by environmental change. These findings demonstrate that cameras can enhance robustness and adaptability without continuous monitoring, supporting privacy-preserving deployment in real homes.

## Repository layout (deployment-side)
- [`deployment/main_deployment.py`](deployment/main_deployment.py): Entry point for running the full deployment pipeline.
- [`deployment/config/config_deployment.py`](deployment/config/config_deployment.py): Deployment configuration (paths, toggles, thresholds).
- [`deployment/sensor_inference.py`](deployment/sensor_inference.py): Non-vision inference wrapper.
- [`deployment/uncertainty/confidence_estimation.py`](deployment/uncertainty/confidence_estimation.py): Confidence estimation and calibration.
- [`deployment/uncertainty/fallback_manager.py`](deployment/uncertainty/fallback_manager.py): Vision fallback activation for low-confidence or sensor-error cases.
- [`deployment/drift_detection/baseline_manager.py`](deployment/drift_detection/baseline_manager.py) and [`deployment/drift_detection/drift_detector.py`](deployment/drift_detection/drift_detector.py): Baseline maintenance and drift detection.
- [`deployment/retraining/retraining_manager.py`](deployment/retraining/retraining_manager.py): Automatic retraining coordinator.
- Logs and outputs: `logs/<use_case>/decisions.csv`, `logs/<use_case>/drift_history.csv`, `alerts/`.

## Prerequisites
- Python 3.9+ and required dependencies (install via your environment or `pip install -r requirements.txt` if provided).
- Trained models and data paths configured in [`DeploymentConfig`](deployment/config/config_deployment.py).
- Incoming sensor CSV and image timestamp CSV paths set per use case in `DeploymentConfig.use_case_config`.
- Vision and sensor models placed under `deployment/models/<use_case>/`.

## Quickstart (deployment)
1. Activate your environment and set working directory to repository root.
2. Run the deployment pipeline for a chosen use case (e.g., `door`):
   ```bash
   python deployment/main_deployment.py --use_case door
   ```
3. Outputs:
   - Decisions/logs: `logs/door/decisions.csv`
   - Alerts: `alerts/`
   - Drift history (if enabled): `logs/door/drift_history.csv`

## Pipeline stages (toggled in [`main_deployment.py`](deployment/main_deployment.py))
1. **Sensor inference** (non-vision): [`run_sensor_model`](deployment/sensor_inference.py) produces predictions/logits.
2. **Confidence estimation**: [`estimate_confidence`](deployment/uncertainty/confidence_estimation.py) calibrates scores and marks low-confidence rows.
3. **Vision fallback**: [`vision_fallback`](deployment/uncertainty/fallback_manager.py) runs vision only for low-confidence or sensor-error cases.
4. **Drift detection** (optional): [`detect_drift`](deployment/drift_detection/drift_detector.py); baseline maintenance via [`update_baseline`](deployment/drift_detection/baseline_manager.py).
5. **Auto-retraining** (optional): [`RetrainingManager.trigger_retraining`](deployment/retraining/retraining_manager.py) when drift is flagged.

To adjust which stages run, set the toggles in [`main_deployment.py`](deployment/main_deployment.py) (e.g., `config.enable_drift_detection`, `config.enable_auto_retraining`).

## Notes on configuration
- Use-case-specific paths (sensor CSV, image folder, timestamps CSV, models) are resolved in [`DeploymentConfig`](deployment/config/config_deployment.py). Ensure `use_case_config` entries are correct.
- Drift/retraining thresholds: `config.drift_threshold`, `config.performance_threshold`, `config.min_samples_for_retraining`.
- Vision fallback thresholds: see [`fallback_manager.py`](deployment/uncertainty/fallback_manager.py) for confidence cutoffs and matching rules.

## Training pipeline (high level)
- Self-supervised vision clustering and pseudo-labeling: see [`training/ssl_subsystem`](training/ssl_subsystem) modules (e.g., `ssl_train_cluster.py` and `old_versions` for experimental variants).
- Label refinement combining vision + sensors: [`training/label_refinement_subsystem/refine_labels.py`](training/label_refinement_subsystem/refine_labels.py).
- Non-vision anomaly training and delay calibration: [`training/non_vision_subsystem`](training/non_vision_subsystem) (e.g., `detect_anomalies.py`, `dynamic_delay_calibration.py`).

## Typical workflow
1. Prepare data and train sensor + vision models using the training scripts.
2. Place trained artifacts under `deployment/models/<use_case>/`.
3. Configure paths/toggles in [`config_deployment.py`](deployment/config/config_deployment.py).
4. Run the deployment pipeline.
5. Monitor `logs/<use_case>/decisions.csv`, drift history, and alerts; enable auto-retraining if desired.

## Support
For issues with deployment stages, check the printed logs from `main_deployment.py`, the decision log under `logs/<use_case>/`, and module-specific outputs in `alerts/` and `logs/<use_case>/`.