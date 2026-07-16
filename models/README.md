# Model file

Place the trained Keras model here with this exact name:

```text
models/best_model.keras
```

The expected output class order is:

```text
glioma, meningioma, notumor, pituitary
```

This order comes from `image_dataset_from_directory`, which sorts folder names alphabetically. If your exported model uses another class order, update the `CLASS_NAMES` environment variable before deployment.
