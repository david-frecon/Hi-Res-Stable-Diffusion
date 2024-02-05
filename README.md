Auteurs:
- Anthony BERNARD
- David FRECON
- Junyi LI
- Louis PAGNIER
- Léandre PERROT
- Léo SRON

# Hi-Res-Stable-Diffusion

## Description

Le but de ce projet est d'implémenter le papier [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf).
Pour arriver à ce résultat, nous avons dans un premier temps implémenté un modèle de diffusion classique, puis nous avons implémenté et encapsulé ce modèle dans un espace latent pour obtenir un modèle de diffusion latente.
De plus, nous avons ajouté des vecteurs décrivant une description textuelle de l'image pour faire de la génération d'image à partir de texte. L'encodeur du texte est celui de [FashionCLIP](https://github.com/patrickjohncyh/fashion-clip).

## Résultats

Voici un exemple de génération d'images de vêtements à partir de texte avec notre modèle entraîné :
Textes utilisés : "a red dress", "blue T-Shirt", "pink tshirt", "a yellow jean", "black dress".


On peut voir que le modèle est capable de générer des images de vêtement non présentes dans le dataset (comme le jean jaune), ce qui montre que le modèle a réussi à interpoler les différents concepts de vêtements au sein de l'espace latent pour générer des images cohérentes. 

## Installation

Pour installer les dépendances, il suffit de lancer la commande suivante depuis la racine du projet :
```bash
poetry install
```
De plus, voici un lien pour télécharger les poids des modèles utilisés pour l'inférence : https://drive.google.com/file/d/1Z5UbMDq9G8L-gSbGKzfCB2zsWnuwm5oK/view?usp=drive_link  
Il faut que les poids en `.pt` et `.pth` soient à la racine du dossier `./models/` pour lancer les scripts d'inférence sans faire de modifications.

## Utilisation

Pour lancer notre application, il suffit de lancer la commande suivante depuis la racine du projet :
```bash
poetry run python StableDiffusion/app.py
```

Si vous souhaitez lancer directement les scripts d'inférence ou d'entraînement, vous devez d'abord ajouter le dossier à votre `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Entraînement

DDPM:
```bash
poetry run python StableDiffusion/DDPM/train_DDPM.py
```

VAE:
```bash
poetry run python StableDiffusion/VAE/train_VAE.py
```

Stable Diffusion:
```bash
poetry run python StableDiffusion/train.py
```

### Inférence

DDPM:
```bash
poetry run python StableDiffusion/DDPM/run_DDPM.py
```

VAE:
```bash
poetry run python StableDiffusion/VAE/run_VAE.py
poetry run python StableDiffusion/VAE/plot_latent_space.py # Pour visualiser l'espace latent
```

Stable Diffusion:
```bash
poetry run python StableDiffusion/inference.py
```

