# Cow Recognition Project / Projekt Rozpoznawania Krów
![Logo](https://th.bing.com/th/id/OIG1.LcAmnQG4R3jxLDWxa2kl?w=1024&h=1024&rs=1&pid=ImgDetMain)

## Polish Version
### Wprowadzenie
Celem projektu jest rozwiązanie rzeczywistego problemu: rozpoznawanie krów na podstawie ich zdjęć. Zastosowanie tego projektu jest szczególnie przydatne w zarządzaniu dużymi gospodarstwami bydła, gdzie trzeba skutecznie identyfikować i śledzić pojedyncze sztuki bydła. Projekt koncentruje się na opracowaniu modelu uczenia maszynowego w Pythonie, który będzie w stanie dokładnie rozpoznawać krowy w gospodarstwie na podstawie ich zdjęć uchwyconych poprzez kamery znajdujące się w gospodarstwie.

### Motywacja
Dokładne identyfikowanie ras krów jest kluczowe dla zarządzania gospodarstwem, programów hodowlanych oraz zapewnienia zdrowia bydła. Technologia ta może zautomatyzować proces identyfikacji krów, zmniejszając czas i wysiłek rolników oraz poprawiając ogólne zarządzanie stadem. Zastępuje ona znakowanie (chipowanie) bądź samodzielne zliczanie bydła.

### Cele
- **Cel:** Opracowanie modelu uczenia maszynowego do rozpoznawania różnych ras krów występujących w gospodarstwie na podstawie ich zdjęć - biorąc pod uwagę ich umaszczenia.
- **Motywacja:** Zwiększenie efektywności zarządzania gospodarstwem i dokładności w identyfikacji ras krów. Zastąpienie znakowania (chipowania) bądź samodzielnego zliczanie bydła. Śledzenie położenia danego osobnika (np.: poprzez informacje czy jest w budynku oraz w jakim miejscu tego budynku się znajduje).
- **Dane wejściowe:** Zdjęcia krów, wstępnie przetworzone do standardowych rozmiarów w celu trenowania modelu i dokonywania predykcji.

### Dziedzina sztucznej inteligencji
Projekt ten związany jest z widzeniem komputerowym, poddziedziną sztucznej inteligencji skoncentrowaną na umożliwianiu maszynom interpretacji i podejmowania decyzji na podstawie danych wizualnych.

### Zbiór danych
Zbiór danych składa się z oznaczonych zdjęć różnych ras krów. Każde zdjęcie jest wstępnie przetwarzane i normalizowane przed wprowadzeniem do modelu.

### Metodologia
**1. Ładowanie i przetwarzanie zbioru danych:** Zdjęcia są zmniejszane do rozmiaru 128x128 pikseli i normalizowane.

**2. Budowa modelu CNN:** Konwolucyjna sieć neuronowa (CNN) jest konstruowana przy użyciu Keras z warstwami dla konwolucji, poolingu, oraz dropoutu.

**3. Trenowanie modelu:** Model jest trenowany z użyciem zaugmentowanych danych obrazowych, aby zwiększyć jego odporność.

**4. Ewaluacja modelu:** Wydajność modelu jest oceniana za pomocą metryk klasyfikacyjnych.

**5. Zapis modelu:** Wytrenowany model

**6. Przewidywanie nowych obrazów:** Model służy do przewidywania pojedynczych krów na nowych obrazach.

## English Version
### Introduction
This project aims to solve a real-world problem: identifying cows based on their images. The application of this project is particularly useful in managing large cattle farms where different breeds of cows need to be identified and tracked efficiently. The project focuses on developing a machine learning model using Python that can accurately recognize different breeds of cows from images.

### Motivation
The ability to accurately identify cow breeds is essential for farm management, breeding programs, and ensuring the health of livestock. This technology can automate the process of cow identification, reducing the time and effort required by farmers and improving the overall management of the herd.

### Objectives
- **Goal:** Develop a machine learning model to recognize different breeds of cows from images.
- **Motivation:** Improve farm management efficiency and accuracy in identifying cow breeds.
- **Inputs:** Images of cows, preprocessed to standard sizes for model training and prediction.

### Artificial Intelligence Domain
This project is related to computer vision, a subfield of artificial intelligence focused on enabling machines to interpret and make decisions based on visual data.

### Dataset
The dataset consists of labeled images of various cow breeds. Each image is preprocessed and normalized before being fed into the model.

### Methodology
**1. Load and preprocess dataset:** Images are resized to 128x128 pixels and normalized.

**2. Build a CNN model:** A Convolutional Neural Network (CNN) is constructed using Keras with layers for convolution, pooling, and dropout.

**3. Train the model:** The model is trained using augmented image data to enhance its robustness.

**4. Evaluate the model:** The model's performance is assessed using classification metrics.

**5. Save the model:** The trained model is saved for future predictions.

**6. Predict new images:** The model is used to predict the breed of cows in new images.
