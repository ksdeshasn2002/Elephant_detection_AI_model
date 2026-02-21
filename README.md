# Edge AI Vision Model Training Pipeline

This repository contains the complete Machine Learning pipeline used to train, augment, optimize, and quantize a lightweight image classification model for edge devices. 

While the architecture and scripts here can be adapted for various edge AI tasks, this specific pipeline was developed to train the vision model for **Elefence**—an IoT prototype designed to detect elephants at agricultural borders in Sri Lanka using an ESP32 microcontroller.



---

### The Data Challenge: Fine-Tuning & Transfer Learning
A major hurdle in training vision models for specialized environments (such as jungle borders) is the lack of large, high-quality, domain-specific datasets. 

To achieve a viable detection rate without requiring tens of thousands of images, this pipeline utilizes **Transfer Learning (Fine-Tuning)**. Instead of training a Convolutional Neural Network (CNN) from scratch, we imported a pre-trained base model. By freezing the early layers (which already understand basic shapes, edges, and textures) and only training the final classification layers on our limited dataset, we achieved a strong baseline accuracy that would have otherwise been impossible given the sample size constraints.

---

### Synthetic Night-Vision Augmentation
For continuous 24/7 monitoring, the model must perform accurately at night. However, sourcing real infrared (IR) or night-vision images of specific targets (like wild elephants) is highly difficult. 

To overcome this, we implemented a custom data augmentation pipeline that mathematically mimics night-time camera feeds. During the training process, standard daylight images are dynamically altered using the following techniques:
1. **Grayscale Conversion:** Color channels are stripped to simulate the monochromatic output of an IR camera.
2. **Brightness & Contrast Shifting:** Pixel values are scaled down to simulate low-light environments and poor illumination.
3. **Noise Injection:** Artificial Gaussian grain is added to simulate the heavy sensor noise typical of budget microcontroller cameras (such as the OV2640).

This synthetic augmentation forces the model to rely on silhouettes and structural shapes rather than color or high-resolution textures.

---

### Model Optimization & Quantization
Deploying a neural network to a microcontroller with highly constrained memory (like an ESP32-S3) requires aggressive optimization. The raw model was processed using the TensorFlow Lite Converter with the following specifications:

* **Full Integer Quantization:** Internal weights and activations were scaled down from 32-bit floating-point (`Float32`) to 8-bit integers (`INT8`). This drastically reduces the memory footprint and speeds up inference on edge hardware.
* **I/O Type Matching:** The input and output tensors were explicitly kept as `Float32`. This was a deliberate engineering choice to maintain compatibility with standard C++ image preprocessing functions on the microcontroller, requiring specific `AddQuantize()` and `AddDequantize()` layers during deployment.
* **Final Footprint:** The optimized `.tflite` model was successfully reduced to approximately **2.8MB**, allowing it to be loaded entirely into the PSRAM of the target device.

---

### Current Results & Limitations
Despite the limited dataset, the fine-tuned model demonstrates reliable detection under controlled lighting and clear line-of-sight conditions. 

However, due to the lack of diverse real-world samples, the model currently exhibits vulnerabilities to false positives in highly complex environments (e.g., dense foliage or shadows that mimic animal silhouettes). Furthermore, while the synthetic night-vision augmentation significantly improved low-light performance, it is not a perfect 1:1 substitute for native thermal or IR imagery.

---

### Future Improvements
To push the model's accuracy and robustness further, future iterations of this pipeline will focus on:
1. **Real Night Data Integration:** Replacing synthetic night-vision data with actual infrared captures from the deployment environment.
2. **Hard Negative Mining:** Curating a specific dataset of images that frequently confuse the model (e.g., large boulders, specific tree formations, or other large animals like water buffalo) to explicitly train the model on what *not* to detect.
3. **Hyperparameter Tuning for Size:** Experimenting with the width multiplier (alpha) of the base architecture to shrink the model size below 2MB, which would free up critical heap memory on the target microcontroller for faster network transmissions.

---

### Usage
* Ensure you have TensorFlow 2.x installed.
* Run the Jupyter Notebook `project_code.ipynb` to execute the data pipeline, initiate transfer learning, and output the quantized `.tflite` file.
