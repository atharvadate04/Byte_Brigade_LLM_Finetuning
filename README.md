# Byte_Brigade_LLM_Inference-Finetuning

This repository contains a Jupyter Notebook for inferencing on a chatbot as part of a larger server project with many folders and libraries.
It also contains a Jupyter notebook for finetuning of meta-llama/Llama-2-7b-chat-hf model using custom alpaca_cleaned_dataset from [alpaca_lora](https://github.com/tloen/alpaca-lora) repository.

## Introduction
Part 1: This notebook demonstrates the process of running inferences on a chatbot model. The aim is to showcase the model's capabilities and provide insights into its performance and application.
Part 2: This notebook demonstrates the process of fine-tuning a model on a specific dataset. The dataset was initially too large, so it was reduced to optimize the training time. The notebook provides a step-by-step guide on fine-tuning and evaluating the model. We use 3 kinds of dataset for Text Generation, Summarization and Code Generation. 

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.x
- You have installed Jupyter Notebook
- You have the necessary libraries installed (as listed in the notebook)

## Installation

1. Clone the repo
    ```sh
    git clone https://github.com/intel/intel-extension-for-transformers.git
    ```

2. Navigate to the project directory
    ```sh
    cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/
    ```

3. Install the required libraries
    ```sh
    pip install -r requirements.txt
    ```
    ```sh
    pip install -r requirements_cpu.txt
    ```

## Inference Notebook
![WhatsApp Image 2024-07-13 at 21 12 40_d3e1ab10](https://github.com/user-attachments/assets/dd11ec30-16e3-44a5-84d9-7f48f3b6a537)
![WhatsApp Image 2024-07-13 at 21 16 26_20fe7b7d](https://github.com/user-attachments/assets/3792d8c3-8e10-43d4-9f3d-56e0dcebc462)
![WhatsApp Image 2024-07-14 at 17 01 44_e97a0d58](https://github.com/user-attachments/assets/9c4ef8c3-80ee-4ae6-8dbb-6393849578ee)
![WhatsApp Image 2024-07-14 at 17 09 52_3f13dd07](https://github.com/user-attachments/assets/49e1e2ed-4e8c-4f85-9976-a74a55860a8f)


## Learnings
- We learned that if there are any conflicts with the environment in which you are working, it is necessary to create a virtual environment (using the venv Python library) to resolve module not found errors. You also have to create a custom Kernel to execute the commands in notebooks
- We have learned how to use NeuralChat, a customizable chat framework, to create a chatbot within minutes on various architectures, specifically on the 4th Generation of Intel® Xeon® Scalable Processors (Sapphire Rapids). By leveraging the intel_extension_for_transformers library, We explored how to optimize chatbot performance using BF16 mixed precision. This involved configuring a chatbot with PipelineConfig and MixedPrecisionConfig to enhance computational efficiency and reduce latency, thereby improving the overall user experience in text-based interactions.
-**Model Accuracy and Performance**:
Evaluating how accurately the model performs on new, unseen data.
Understanding the model's strengths and weaknesses in different scenarios.
-**Response Time and Efficiency**:
Measuring the time taken for the model to generate responses.
Analyzing the efficiency of the model in real-time applications.
-**Error Analysis**:
Identifying common types of errors made by the model.
Understanding why these errors occur and how they can be mitigated.

### Future Work

- **Scalability**: Implementing the model in a scalable server environment.
- **Real-time Applications**: Enhancing the model to handle real-time user inputs more efficiently.


## Finetuning Notebook

### Dataset
The dataset used for fine-tuning was cloned from a GitHub repository and then reduced using a python function to optimize training time.

### Images
![image](https://github.com/user-attachments/assets/c19a6c2b-9014-4f1f-9ea1-8ed2d4f9aa98)
![Screenshot (49)](https://github.com/user-attachments/assets/b919daab-f32d-41e5-ab4c-b0bfb0c3158d)
![Screenshot (50)](https://github.com/user-attachments/assets/60d1f2d6-842e-4221-8da7-fab79a9e6948)
![Screenshot (51)](https://github.com/user-attachments/assets/a72a6e1f-5c29-4987-bf5c-cf16fca9f28c)

### Learnings and Challenges
- We have learned how to fine-tune a language model on a single node Xeon SPR using the Intel Extension for Transformers and the Alpaca dataset. The process involved configuring model arguments, data arguments, and training arguments using the Hugging Face transformers library. Additionally, We encountered and addressed various warnings and errors related to TensorFlow, CUDA, and dataset loading. This experience has deepened my understanding of model fine-tuning, dataset handling, and leveraging specialized hardware and software extensions to optimize training performance.
- Also, we got to experience how various parameters such as epochs, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, and save strategy can impact the model training process.
- We got to know about how model compatibility is must while finetuning a dataset.

### Future Work
- We have to work on even larger custom datasets for code generation and summarization tasks.
- Additionally, exploring techniques for incorporating domain-specific knowledge or structured data into the fine-tuning process could potentially improve the models' adaptability and effectiveness in real-world applications.
- Moreover, investigating methods to mitigate biases and enhance model fairness in language generation tasks remains a critical area for future research and development.
  
## Contributing

If you want to contribute to this project, please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgements
- Special thanks to Intel for giving us this opportunity.
- Thanks to [meta-llama](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) for providing the Llama-2 model.
- Thanks to [alpaca_lora](https://github.com/tloen/alpaca-lora) for providing finetuning dataset.
