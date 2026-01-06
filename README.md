# 👕 Fashion MNIST Clothing Classifier

A deep learning project that classifies clothing images into different categories using a **Convolutional Neural Network (CNN)** trained on the **Fashion-MNIST dataset**.  
The project is deployed as an interactive **Gradio web app** on **Hugging Face Spaces**.

🔗 **Live Demo:**  
👉 https://huggingface.co/spaces/harshitash11/Clothes_prediction_model

---

## 📌 Project Overview

This project demonstrates how CNNs can be used for image classification tasks.  
Users can upload a clothing image, and the model predicts the corresponding clothing category along with a confidence score.

The model is trained on **Fashion-MNIST**, which consists of grayscale images of clothing items such as shirts, trousers, shoes, and bags.

---

## 🧠 Model Details

- **Dataset:** Fashion-MNIST
- **Image Size:** 28 × 28 (grayscale)
- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Deployment:** Gradio + Hugging Face Spaces

### Clothing Classes
The model predicts one of the following 10 classes:

- T-shirt / Top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle Boot  

---

## 🚀 Live Application

The application is deployed and publicly accessible:

👉 **Hugging Face Space:**  
https://huggingface.co/spaces/harshitash11/Clothes_prediction_model

### How to Use
1. Open the live link
2. Upload a clothing image
3. Click **Submit**
4. View the predicted class and confidence score

---

## ⚠️ Important Note (Model Limitation)

This model is trained on **Fashion-MNIST**, which contains:
- Low-resolution (28×28)
- Grayscale
- Centered clothing images

As a result:
- The model performs best on **Fashion-MNIST–like images**
- Predictions on **real-world product photos** may be inaccurate due to **domain shift**

This limitation is expected and demonstrates an important real-world ML concept.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Gradio
- Hugging Face Spaces

---

## 📁 Project Structure


---

## 📈 Learning Outcomes

- Understanding CNNs for image classification
- Difference between FCNN and CNN
- Model training vs inference separation
- Handling domain shift
- Deploying ML models using Gradio
- Hosting models on Hugging Face Spaces

---

## 🧪 Future Improvements

- Train on real-world clothing datasets
- Use transfer learning (MobileNet / ResNet)
- Support RGB and higher-resolution images
- Show top-3 predictions with probabilities
- Improve preprocessing for real images

---

## 👩‍💻 Author

**Harshita Sharma**  
AI / ML Enthusiast  

---

⭐ If you like this project, feel free to star the repository and try the live demo!
