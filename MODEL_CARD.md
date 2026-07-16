# Model card — Brain Tumor MRI CNN

## Intended use

This model is an academic proof of concept for four-class brain MRI image classification. It is intended for coursework demonstration, model evaluation, and explainable-AI research.

It is **not a clinical diagnostic device**. Predictions must not be used to make treatment decisions or replace review by qualified medical professionals.

## Architecture

- Input: RGB image resized to 224 × 224
- Backbone: ImageNet-pretrained MobileNetV2 (`include_top=False`)
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(4, softmax)
- Training: frozen feature extraction followed by partial fine-tuning
- Explainability: Grad-CAM from the last convolutional layer

## Class order

1. glioma
2. meningioma
3. notumor
4. pituitary

## Evaluation recorded in the project notebook

- Held-out test images: 1,600 (400 per class)
- Accuracy: 87.81%
- Macro precision: 0.8819
- Macro recall: 0.8781
- Macro F1: 0.8755

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9452 | 0.7325 | 0.8254 | 400 |
| Meningioma | 0.8064 | 0.8225 | 0.8144 | 400 |
| No tumour | 0.9032 | 0.9800 | 0.9400 | 400 |
| Pituitary | 0.8728 | 0.9775 | 0.9222 | 400 |

## Important limitations

- The dataset combines images from multiple sources, scanners, viewpoints, and resolutions.
- External clinical validation has not been performed.
- A high-confidence prediction can still be wrong.
- Grad-CAM is an explanation aid, not proof that the model reasoned clinically.
- Uploaded images should be de-identified before use.
- A production medical system would require security controls, audit logging, monitoring, regulatory review, and clinician oversight.
