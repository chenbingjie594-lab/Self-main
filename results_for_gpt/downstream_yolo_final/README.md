# Frozen DWBG-v2 downstream YOLO11s results

Each subdirectory is one completed YOLO11s run. The `s422` folders are a
second run created by Ultralytics after the requested `s42` name already
existed.

Included per run: epoch metrics (`results.csv`), configuration (`args.yaml`),
normalized confusion matrix, Box PR curve, and Box F1 curve. Model weights,
training-batch previews, and datasets are deliberately excluded.
