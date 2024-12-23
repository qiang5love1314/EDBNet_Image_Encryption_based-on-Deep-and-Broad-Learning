# EDBNet_Image_Encryption_based-on-Deep-and-Broad-Learning
The paper titled "Enhancing Patient Privacy in IoT-Enabled Intelligent Systems: A Deep and Broad Learning-Based Efficient Encryption Network" was accepted by the 2024 IEEE SmartIoT.

## Abstract  
The rapid development of the Internet of Things (IoT) enables a wide range of applications in intelligent medical systems. However, medical imaging equipment often produces sensitive user privacy information, and existing solutions fail to adequately secure these communications. Current cryptographic methods face challenges such as low real-time performance and suboptimal security.  

To address these issues, EDBNet introduces an encryption network based on deep and broad learning, designed specifically to improve the privacy of medical images.  

Key contributions include:  
- Employing a **four-layer convolutional neural network (CNN)** to extract horizontal and vertical features.  
- Leveraging **broad learning** to guide the training process and generate feature matrices.  
- Integrating **chaotic cryptography** with scrambling, diffusion, and the **SHA3-256 algorithm** for enhanced ciphertext security.  
- Utilizing dual-stream encryption with the **COVID-CT-Dataset** for training and fine-tuning.  

Experimental results demonstrate that EDBNet achieves state-of-the-art performance in terms of cipher entropy, encryption quality, and real-time processing.  

## Features  
- **Deep and Broad Learning Framework**: Combines CNNs and broad learning to generate robust encryption features.  
- **Chaotic Cryptography**: Enhances encryption with secure scrambling, diffusion, and SHA3-256.  
- **Real-Time Encryption**: Achieves encryption and decryption in approximately 1 second.  
- **High Security and Performance**: Outperforms traditional methods with an average cipher entropy of 7.9971 and encryption quality of 248.  

## Usage  
The key scripts in this project are:  
- `EncryBroNet.py`: Demonstrates the encryption process with medical images, and it also includes the model training process using the COVID-CT-Dataset.  
- `DecryBroNet.py`: Recovers the original medical images from the encrypted data.  

To run the program:  
1. Clone this repository.  
2. Install dependencies listed in `requirements.txt`.  
3. Execute the encryption or decryption script with example data.  

## Results  
Extensive experiments validate the efficiency and security of EDBNet:  
- **High Security**: Achieves an average cipher entropy of **7.9971**, making it highly resistant to attacks.  
- **Encryption Quality**: Scores a high encryption quality of **248**, ensuring robust protection for medical images.  
- **Real-Time Performance**: Both encryption and decryption complete within **1 second**, enabling practical use in IoT-based medical systems.  

## Citation  
If you find this work helpful, please cite:  
```bibtex  
@inproceedings{zhu2024enhancing,
  title={Enhancing Patient Privacy in IoT-Enabled Intelligent Systems: A Deep and Broad Learning-Based Efficient Encryption Network},
  author={Zhu, Xiaoqiang and Liu, Jiqiang and Zhang, Dalin and Li, Lingkun and Wang, Nan},
  booktitle={2024 IEEE International Conference on Smart Internet of Things (SmartIoT)},
  pages={51--58},
  year={2024},
  organization={IEEE}
}
