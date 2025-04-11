# MiniML

A lightweight AutoML API for small teams. Upload a CSV and get a model and performance report.

## Usage

1. Build the Docker image:
   ```bash
   docker build -t miniml .
   ```

2. Run the API:
   ```bash
   docker run -p 8000:8000 miniml
   ```

3. POST a CSV and target column to `/train`

## Example

```bash
curl -X POST "http://localhost:8000/train"      -F file=@yourdata.csv      -F target=target_column
```