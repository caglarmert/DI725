There are 6 files in this project baseline starter:
1. captioner: Is an example file on how to batch process images to extract captions
1. gemma: The Gemma language model architecture is built here.
1. siglip: The SigLip vision model architecture resides here.
1. inference: The pre-trained model is used with the inference code.
1. processing_paligemma: Where we combine the vision encoder and language decoder models to form the vision language model.
1. utils: Used to extract weights from a pre-trained model.

We will be using PaliGemma model as our starting point in this project. You need to propose an improvement over this model to fine-tune to the image captioning task. Your proposals on how to improve the model can be on various topics including, but not limited to:
* Architectural improvements
* Efficiency and lighter adaptations
* Inference speed
* Efficient fine-tuning processes

The following limitations come from the SigLip and Gemma models:
* Vision-Language Models (VLM) require clear prompts and instructions to work effectively.
* Creative and complex tasks are not (yet) suitable for VLMs.
* VLMs might not be able to understand nuances, sarcasm or figurative language, due to the inherent complexity of natural languages.
* VLMs are not knowledge bases, they can generate responses based on their training but this can be hallucination or incorrect.
* VLMs depend on statistical patterns of vision and language domains. They might not be able to apply common sense.

PaliGemma is:
* A general pre-trained model suitable for fine-tuning to tasks.
* A solid starting point for adapting a VLM model.

PaliGemma is not:
* A zero-shot model that can be used as-is, there are better models for this task.
* A chatbot with multiple question and answer sessions.

