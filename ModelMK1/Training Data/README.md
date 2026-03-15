Place your training dataset files here (CSV or Parquet format).

When you run the model, it will automatically discover and use the most recently modified file in this folder.

Supported formats:
- .csv (comma-separated values)
- .parquet (Apache Parquet)

Example:
  copy your_data.csv "Training Data/"
  python main.py train-lnn

The model expects the following columns:
- timestamp: datetime in parseable format
- price, close, idx_close, last, or ltp: price value
- high, low, volume: optional OHLV fields
