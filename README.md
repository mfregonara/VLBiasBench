
## VLBiasBench

**VLBiasBench** is a large-scale benchmark dataset composed of high-quality synthetic images, designed to evaluate social biases in Large Vision-Language Models (LVLMs).

---

### Running the Close-Ended Evaluation

Follow the steps below to run the close-ended evaluation for a specific model:

1. **Download the Evaluation Datasets**  
   Download both the close-ended and open-ended datasets from the following link:  
   [Google Drive - VLBiasBench Datasets](https://drive.google.com/drive/folders/1YJx-6zCd506Xbm8rUtELKrRMp6nZbRuV)  
   Place the downloaded contents under the `evaluation/data/` directory.

2. **Set Up Your Python Environment**  
   - Use Python 3.11.  
   - Create and activate a virtual environment:
     ```bash
     python3.11 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

3. **Install CUDA Toolkit (v12.6)**  
   Make sure CUDA Toolkit version 12.6 is installed and properly configured on your system for GPU acceleration.


4. **Download the Model to Evaluate**  
   - Create a model download script in `evaluation/model_download_scripts/`.
   - Use the provided `download_blip_blip2-opt-2.7b.py` as a reference for formatting your script.
   - Run the script with:
        ```bash
        python evaluation/model_download_scripts/<your_script_name>.py
        ```

5. **Set the Working Directory**  
   Change your current working directory to the `evaluation/` folder:
   ```bash
   cd evaluation
   ```

6. **Run the Evaluation**
   Execute the close-ended evaluation with your desired model:

   ```bash
   python run_evaluation.py --model_name <model-name> --dataset_list 'close_ended_dataset'
   ```

---


### Overview🔍

![Overview of the construction of VLBiasBench](./docs/Figure1-final.png)

**Figure 1. Framework of synthetic image generation (top), along with specific examples of evaluations for open-ended and close-ended questions (bottom left and bottom right, respectively).**

**Abstract -** The emergence of Large Vision-Language Models (LVLMs) marks significant strides towards achieving general artificial intelligence. However, these advancements are accompanied by concerns about biased outputs, a challenge that has yet to be thoroughly explored. Existing benchmarks are not sufficiently comprehensive in evaluating biases due to their limited data scale, single questioning format and narrow sources of bias. To address this problem, we introduce VLBiasBench, a comprehensive benchmark designed to evaluate biases in LVLMs. VLBiasBench, features a dataset that covers nine distinct categories of social biases, including age, disability status, gender, nationality, physical appearance, race, religion, profession, social economic status, as well as two intersectional bias categories: race × gender and race × social economic status. To build a large-scale dataset, we use Stable Diffusion XL model to generate 46,848 high-quality images, which are combined with various questions to creat 128,342 samples.
These questions are divided into open-ended and close-ended types, ensuring thorough consideration of bias sources and a comprehensive evaluation of LVLM biases from multiple perspectives. We conduct extensive evaluations on 15 open-source models as well as two advanced closed-source models, yielding new insights into the biases present in these models. 

### Open-ended evaluation

![Open-ended evaluation framework](./docs/open-construction.png)

**Figure 2. Prompt library construction for open-ended question evaluation. Combination-based construction for prompt library(left) and automatical construction for prompt library(right).**

### Close-ended evaluation

![Close-ended evaluation framework](./docs/framework-gai.png)

**Figure 3. Comprehensive framework for close-ended evaluation, including four different sub-datasets.**

### Key statistics of VLBiasBench📊

| Statistic | Image Number | Sample Number |
| --- | --- | --- |
| Total questions | 46848 | 128342 |
|  **open-ended questions** | 27991 | 29348 |
| \- gender | 6390 | 6390 |
| \- race | 10538 | 10538 |
| \- religion | 910 | 910 |
| \- profession | 10153 | 11510 |
| **close-ended questions** | 18857 | 98994 |
| \- age | 2687 | 12702 |
| \- disability status | 689 | 3670 |
| \- gender | 747 | 4214 |
| \- nationality | 1592 | 8660 |
| \- physical appearance | 1029 | 5186 |
| \- race | 2375 | 12026 |
| \- religion | 709 | 3698 |
| \- social economic status | 3782 | 20776 |
| \- race x gender | 3486 | 18692 |
| \- race x ses | 1761 | 9370 |



### Examples of VLBiasBench📸

![A sample in open-ended evaluation](./docs/Open_African.png)

![A sample in close-ended evaluation](./docs/Close_3n.png)

<!--### Evaluation Results🏆

![Overview of the construction of VLBiasBench](./docs/Figure2.png)
**Figure 4. Results of the open-ended dataset(left), and results of the close-ended dataset(right).**-->

### Related projects🔗

+   [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
+   [EMU2](https://github.com/baaivision/Emu)
+   [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip)
+   [LLaVA-1.5](https://github.com/haotian-liu/LLaVA)
+   [miniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)
+   [miniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)
+   [Otter](https://github.com/Vision-CAIR/MiniGPT-4)
+   [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
+   [Shikra](https://github.com/shikras/shikra)
+   [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)

### Cite our work📝

@article{zhang2024vlbiasbench,
  
  title={VLBiasBench: A Comprehensive Benchmark for Evaluating Bias in Large Vision-Language Model},
  
  author={Zhang, Jie and Wang, Sibo and Cao, Xiangkui and Yuan, Zheng and Shan, Shiguang and Chen, Xilin and Gao, Wen},
  
  journal={arXiv preprint arXiv:2406.14194},
  
  year={2024}

}      
    

### Acknowledgement

This section will include acknowledgements...

### Links

🔗The link to our project is [\[github\]](https://github.com/Xiangkui-Cao/VLBiasBench)

🔗The link to our paper is [\[arxiv\]](https://arxiv.org/abs/2406.14194)

🔗The link to our dataset is [\[Google Drive\]](https://drive.google.com/drive/folders/1YJx-6zCd506Xbm8rUtELKrRMp6nZbRuV?usp=drive_link)

