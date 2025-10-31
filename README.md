# Waste Classification & Segmentation

Projet de **vision par ordinateur** pour la **classification** et la **segmentation** des déchets.  
Construit avec **TensorFlow / Keras** et déployé via **Streamlit** pour une interface interactive.

---

## Features

- **Classification** : identifier le type de déchet à partir d’une image.
- **Segmentation** : détecter et segmenter les déchets dans l’image.
- **Interface Streamlit** : pour tester facilement le modèle sur des images locales ou via webcam.
- Supporte plusieurs architectures pré-entraînées pour la classification (MobileNetV2, ResNet50, InceptionV3).
- Optimisation des performances grâce au **preprocessing**, **augmentation d’images** et **caching**.

---

## Installation

Installez les packages nécessaires :

```bash
pip install streamlit tensorflow opencv-python Pillow numpy matplotlib
```


#Dataset

Classification : nous avons utilisé le dataset standard de Kaggle Waste Classification
.

Segmentation : nous avons travaillé avec le dataset ZeroWaste Segmentation
 pour entraîner le modèle de segmentation.

Pour des raisons de taille, les datasets ne sont pas inclus dans le repo GitHub.
Les utilisateurs doivent télécharger les datasets et les placer dans data/ en respectant la structure ci-dessous.
