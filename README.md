# ClearPixAI

ClearPixAI is a watermark removal tool built as a two-stage pipeline:
- **Detection**: a segmentation model predicts a watermark mask
- **Inpainting**: a diffusion inpainting backend restores the masked area

This repository focuses on the **training pipeline for the watermark detector**.

## Overview

- **Task**: binary segmentation (watermark vs background)
- **Output**: a mask used by the watermark removal pipeline
- **Training**: config-driven (`configs/train_config.yaml`)

## Metrics (targets)

| Metric | Target | Minimum acceptable |
|--------|--------|-------------------|
| **IoU** | ≥ 0.90 | ≥ 0.80 |
| **Dice (F1)** | ≥ 0.90 | ≥ 0.80 |
| **Precision** | ≥ 0.85 | ≥ 0.75 |
| **Recall** | ≥ 0.85 | ≥ 0.75 |

Quality gate (recommended): **IoU ≥ 0.80**.

## Data description

The dataset is made of **paired images** (watermarked + clean). Masks are generated automatically by computing the difference between the pair.

### Directory layout

```text
data/
├── clean/
│   ├── image001.jpg
│   └── ...
└── watermarked/
    ├── image001_v1.jpg
    └── ...
```

Validate your pair dataset before training:

```bash
uv run python clearpixai/training/detector/validate_data.py --data-dir /path/to/data
```

## Data Versioning with DVC

This project uses [DVC (Data Version Control)](https://dvc.org/) to manage large datasets and model files outside of Git.

### Physical Storage

- **Datasets**: Stored in DVC cache (`.dvc/cache/`) and remote storage
  - Training data: `clearpixai/training/detector/data/train/`
  - Validation data: `clearpixai/training/detector/data/val/`
- **Models**: Stored in DVC cache and remote storage
  - Exported models: `exported_models/`

### Getting Started with DVC

After cloning the repository:

```bash
cd ClearPixAi
dvc pull
dvc repro
```

This will:
1. Download all datasets and models from remote storage
2. Run the complete pipeline (prepare → train → evaluate)

### DVC Pipeline

The project includes a DVC pipeline with three stages:

1. **prepare**: Validates and prepares the dataset
2. **train**: Trains the watermark detection model
3. **evaluate**: Validates the trained model and exports it

Run individual stages:

```bash
dvc repro prepare    # Prepare data only
dvc repro train      # Train model (requires prepare)
dvc repro evaluate   # Evaluate model (requires train)
```

### Experiment Plan

The DVC pipeline allows you to:
- Track different versions of datasets and models
- Reproduce any experiment with `dvc repro`
- Compare metrics across experiments using `dvc metrics show`
- Switch between dataset/model versions using Git tags and `dvc checkout`

## Experiment Tracking with MLflow

This project uses [MLflow](https://mlflow.org/) to track training experiments, parameters, metrics, and artifacts.

### MLflow Setup

MLflow is automatically configured to use **local storage** (`./mlruns/` directory). Each training run creates a new experiment run with:

- **Parameters**: All training configuration (learning rate, batch size, encoder name, etc.)
- **Metrics**: Training and validation metrics (loss, IoU) logged during training
- **Artifacts**:
  - Training configuration file (`configs/train_config.yaml`)
  - DVC lock file (`dvc.lock`) for reproducibility
  - Best model checkpoint
  - TensorBoard logs
  - Saved training config copy

### Viewing Results

After training, you can view experiment results using the MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open your browser to `http://localhost:5000` to:
- Compare different runs
- View parameters and metrics
- Download artifacts
- Search and filter experiments

### MLflow Storage Location

All MLflow data is stored locally in:
- **Tracking URI**: `file:./mlruns`
- **Experiment name**: `clearpixai_watermark_detector` (or as specified in config)

Each run is automatically tracked when you execute:

```bash
uv run python clearpixai/training/detector/train_from_config.py \
  --config configs/train_config.yaml
```

The MLflow run ID is displayed in the training logs, and you can use it to reference specific runs.

### What Gets Logged

- **Hyperparameters**: All training parameters from config file
- **Metrics**: `train_loss`, `val_loss`, `train_iou`, `val_iou`, `best_val_iou`, `total_epochs`
- **Artifacts**: Config files, checkpoints, TensorBoard logs, DVC lock file
- **Model**: PyTorch Lightning model (via autolog)

## Run instructions

For a quick walk-through, see `QUICKSTART.md`. Below is the short version.

### Install

```bash
uv sync
```

or:

```bash
pip install -r requirements.txt
```

**Note**: If using DVC, also install DVC:

```bash
pip install dvc
# or
uv add dvc
```

### Configure training

Edit `configs/train_config.yaml` and set:
- `data.data_dir`: your dataset path
- `data.batch_size`: reduce if you hit OOM

### Train

```bash
uv run python clearpixai/training/detector/train_from_config.py \
  --config configs/train_config.yaml \
  --verbose
```

Optional (monitor training):

```bash
tensorboard --logdir checkpoints
```

View MLflow experiments:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Validate

```bash
uv run python clearpixai/training/detector/validate.py \
  --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
  --data-dir /path/to/validation/data \
  --output validation_metrics.json
```

### Export

```bash
uv run python clearpixai/training/detector/export_model.py \
  --checkpoint checkpoints/watermark-epoch=XX-val_iou=0.XXXX.ckpt \
  --output-dir exported_models/my_model \
  --model-name "my-watermark-detector-v1"
```

### Run inference (CLI)

```bash
uv run clearpixai -i input.jpg -o output_clean.jpg
```

Use your exported weights:

```bash
uv run clearpixai -i input.jpg -o output_clean.jpg \
  --segmentation-weights exported_models/my_model/pytorch_model.pth
```

## Docker Deployment

ClearPixAI can be packaged as a Docker container for reproducible offline inference.

### Building the Docker Image

**Prerequisites:** Make sure Docker Desktop (or Docker daemon) is running before building.

Build the Docker image:

```bash
docker build -t ml-app:v1 .
```

Or using Docker Buildx (recommended):

```bash
docker buildx build -t ml-app:v1 .
```

This will:
- Install all dependencies from `requirements.txt`
- Copy the source code (`clearpixai/` package)
- Copy the default segmentation model weights
- Set up the entrypoint for inference

**Troubleshooting:**
- If you see "Cannot connect to the Docker daemon", make sure Docker Desktop is running
- On macOS, start Docker Desktop from Applications
- The legacy builder is deprecated; consider using `docker buildx` instead

### Running Inference in Docker

Run the container with input/output files mounted:

```bash
docker run --rm \
  -v /path/to/input/image.jpg:/app/input.jpg:ro \
  -v /path/to/output:/app/output \
  ml-app:v1 \
  --input_path /app/input.jpg \
  --output_path /app/output/cleaned.jpg
```

Or using relative paths:

```bash
docker run --rm \
  -v $(pwd)/input.jpg:/app/input.jpg:ro \
  -v $(pwd):/app/output \
  ml-app:v1 \
  --input_path /app/input.jpg \
  --output_path /app/output/cleaned.jpg
```

### Docker Script Details

The Docker container uses `src/predict.py` as the entrypoint, which:

- **Loads the model**: Automatically loads the segmentation model from disk (default: `clearpixai/detection/best_watermark_model_mit_b5_best.pth`)
- **Accepts arguments**:
  - `--input_path` / `-i`: Path to input image file (required)
  - `--output_path` / `-o`: Path to save output cleaned image (required)
  - `--model_path` or `--segmentation-weights` / `-w`: Optional path to custom segmentation model weights (inside container)
  - `--verbose` / `-v`: Enable verbose logging
- **Input format**: Supports common image formats (JPG, JPEG, PNG, etc.)
- **Output format**: Saves cleaned image as JPG with quality=95

**Совместимость с CLI**: Скрипт поддерживает те же аргументы, что и `clearpixai` CLI, включая `--segmentation-weights` / `-w`.

### Using Custom Model Weights

**Нет необходимости пересобирать Docker образ!** Просто смонтируйте модель при запуске контейнера.

Пример использования кастомной модели (аналог `uv run clearpixai -i test_image.jpg -o test_image_cleaned.jpg --segmentation-weights exported_models/my_model_v3/pytorch_model.pth`):

```bash
docker run --rm \
  -v $(pwd)/test_image.jpg:/app/input.jpg:ro \
  -v $(pwd)/exported_models/my_model_v3/pytorch_model.pth:/app/model.pth:ro \
  -v $(pwd):/app/output \
  ml-app:v1 \
  --input_path /app/input.jpg \
  --output_path /app/output/test_image_cleaned.jpg \
  --segmentation-weights /app/model.pth
```

Или используя короткий вариант `-w`:

```bash
docker run --rm \
  -v $(pwd)/test_image.jpg:/app/input.jpg:ro \
  -v $(pwd)/exported_models/my_model_v3/pytorch_model.pth:/app/model.pth:ro \
  -v $(pwd):/app/output \
  ml-app:v1 \
  -i /app/input.jpg \
  -o /app/output/test_image_cleaned.jpg \
  -w /app/model.pth
```

**Как это работает:**
1. `-v $(pwd)/exported_models/my_model_v3/pytorch_model.pth:/app/model.pth:ro` — монтирует файл модели с хоста в контейнер по пути `/app/model.pth` (только для чтения)
2. `--segmentation-weights /app/model.pth` — указывает путь **внутри контейнера**, где находится смонтированная модель
3. Образ не нужно пересобирать — модель подключается динамически при запуске

**Важно**: 
- Путь к модели в аргументе (`/app/model.pth`) — это путь **внутри контейнера**
- Путь в `-v` — это путь на **хосте** (`$(pwd)/exported_models/...`)
- Docker автоматически связывает эти пути при монтировании

### GPU Support

For GPU acceleration, use `--gpus all`:

```bash
docker run --rm --gpus all \
  -v /path/to/input/image.jpg:/app/input.jpg:ro \
  -v /path/to/output:/app/output \
  ml-app:v1 \
  --input_path /app/input.jpg \
  --output_path /app/output/cleaned.jpg
```

Note: The container will automatically detect and use GPU if available.

## TorchServe Deployment

ClearPixAI can be deployed as an online service using [TorchServe](https://pytorch.org/serve/), which provides a REST API for model inference.

### Prerequisites

Install TorchServe and model archiver:

```bash
pip install torchserve torch-model-archiver
```

### Preparing Model Artifacts

1. **Export model to TorchScript format**:

```bash
uv run python export_torchscript.py \
    --weights exported_models/latest/pytorch_model.pth \
    --config exported_models/latest/config.json \
    --output model-store/model.pt \
    --image-size 512
```

2. **Create TorchServe archive**:

```bash
uv run python create_torchserve_archive.py \
    --model model-store/model.pt \
    --handler torchserve_handler.py \
    --config exported_models/latest/config.json \
    --model-name mymodel \
    --output-dir model-store
```

Or use the automated script:

```bash
./prepare_torchserve.sh
```

This will create `model-store/mymodel.mar` - the model archive ready for deployment.

### Building Docker Image

Build the TorchServe Docker image:

```bash
docker build -f Dockerfile.torchserve -t mymodel-serve:v1 .
```

The Dockerfile:
- Uses `pytorch/torchserve:latest` as base image
- Installs required dependencies (segmentation-models-pytorch, PIL, numpy)
- Copies the model archive (`mymodel.mar`) to the model store
- Configures TorchServe with custom `config.properties`
- Exposes ports 8080 (inference) and 8081 (management)

### Running the Container

Start the TorchServe container:

```bash
docker run -d -p 8080:8080 -p 8081:8081 mymodel-serve:v1
```

The service will be available at:
- **Inference API**: `http://localhost:8080`
- **Management API**: `http://localhost:8081`

### Testing the Service

1. **Prepare sample input**:

```bash
uv run python prepare_sample_input.py test_image.jpg -o sample_input.json
```

This creates a JSON file with base64-encoded image data.

2. **Send prediction request**:

```bash
curl -X POST http://localhost:8080/predictions/mymodel -T sample_input.json
```

The response will contain:
- `mask`: Base64-encoded binary mask image (PNG format)
- `watermark_ratio`: Percentage of image covered by watermark (0.0-1.0)
- `max_confidence`: Maximum confidence score from the model
- `threshold`: Threshold used for binarization
- `mask_shape`: Shape of the mask [height, width]

### REST API Endpoints

#### Inference API (Port 8080)

**POST `/predictions/{model_name}`**
- **Request body**: JSON with base64-encoded image
  ```json
  {
    "image": "base64_encoded_image_data",
    "original_size": [width, height]  // optional
  }
  ```
- **Response**: JSON with mask and statistics
  ```json
  {
    "mask": "base64_encoded_mask_image",
    "watermark_ratio": 0.15,
    "max_confidence": 0.95,
    "threshold": 0.5,
    "mask_shape": [512, 512]
  }
  ```

**Example with curl**:
```bash
curl -X POST http://localhost:8080/predictions/mymodel \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

#### Management API (Port 8081)

**GET `/models`** - List all registered models
```bash
curl http://localhost:8081/models
```

**GET `/models/{model_name}`** - Get model information
```bash
curl http://localhost:8081/models/mymodel
```

**PUT `/models?url={model_url}`** - Register a new model
```bash
curl -X PUT http://localhost:8081/models?url=mymodel.mar
```

**DELETE `/models/{model_name}`** - Unregister a model
```bash
curl -X DELETE http://localhost:8081/models/mymodel
```

### Configuration Parameters

The `config.properties` file contains TorchServe configuration:

- `inference_address`: Inference API address (default: `http://0.0.0.0:8080`)
- `management_address`: Management API address (default: `http://0.0.0.0:8081`)
- `num_workers`: Number of worker processes (default: 1)
- `default_workers_per_model`: Workers per model (default: 1)
- `max_request_size`: Maximum request size in bytes (default: 6553500)
- `max_response_size`: Maximum response size in bytes (default: 6553500)

To customize, edit `config.properties` and rebuild the Docker image.

### Handler Details

The `torchserve_handler.py` implements:

**Preprocessing**:
- Accepts images as base64-encoded strings or file paths
- Converts to RGB format
- Resizes to model input size (default: 512x512)
- Applies encoder-specific normalization (ImageNet for mit_b5)
- Converts to tensor format [batch, channels, height, width]

**Postprocessing**:
- Applies sigmoid activation to logits
- Thresholds mask (default: 0.5)
- Resizes mask back to original image size if needed
- Returns base64-encoded mask image and statistics

### Troubleshooting

**Model not found**:
- Ensure the `.mar` file is in `model-store/` directory
- Check that model name in Dockerfile CMD matches the archive name

**Port already in use**:
- Change port mappings: `docker run -d -p 8082:8080 -p 8083:8081 ...`
- Update `config.properties` accordingly

**Out of memory**:
- Reduce `num_workers` in `config.properties`
- Use smaller batch sizes or image sizes

**Handler import errors**:
- Ensure all dependencies are installed in the Docker image
- Check that `torchserve_handler.py` is included in the archive

## Testing

Install dev dependencies (includes `pytest`):

```bash
make install-dev
```

Run tests:

```bash
make test
```

More commands and test categories: `TESTING.md`.