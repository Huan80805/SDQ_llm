# Efficient LLMs via Switchable and Dynamic Quantization


## Project Workflow
1.  **Setup**: Install the required dependencies.
    ```bash
    pip install torch transformers datasets evaluate peft tensorboard tqdm
    ```

2.  **Training**: Run the main training script to fine-tune the quantized GPT-2 model on the generated data. The generated data follows the approach in [LLM-QAT](https://arxiv.org/abs/2305.17888). I Use SQuAD (`context` + `question`) to prompt GPT-2 to generate completions for quantization fine-tuning. Run **`python gen_data.py`**. This will generate text data and save at `squad_gpt_sft/`

3.  **Analysis & Evaluation**: Use the analysis scripts (detailed below) to find optimal quantization configurations and the evaluation script to measure performance.

## Implementation Details (Step by Step)

### Step 1: Per-Layer Quantization Integration

-   **`src/quant_utils.py`**: The `QuantLinear` class is a custom `nn.Linear` layer that simulates quantization (Actual quantization is implemented during inference). It maintains a full-precision (FP32) copy of its weights and, during the `forward` pass, dynamically quantizes them to a specified `bit_width` (supporting 2, 4, and 8 bits). The quantization is a standard symmetric uniform scheme where a scaling factor is calculated per-row. The `set_bit_width` method allows for changing the precision on-the-fly.
-   **`src/modeling_gpt2.py`**: The `patch_gpt2_with_quantization` function traverses the pre-trained GPT-2 model and replaces all `Conv1D` layers (equivalent to linear layers in GPT-2) with `QuantLinear` layers. This makes the entire model quantizable.

### Step 2: Adaptive LoRA for Switchable Precision

To recover performance lost during quantization and to manage different precision settings, LoRA modules are added.

-   **`src/modeling_gpt2.py`**: The `patch_gpt2_with_adaptive_adapters` function takes the quantized model and adds a unique LoRA adapter for *each combination* of a linear layer and a supported bit-width.
-   **`SwitchableQuantLoRAModel`**: This class acts as a wrapper around the PEFT model. Its primary role is to manage the complexity of switching between quantization configurations, activating required adpters while setting correct bit_width in `QuantLinear`. The `set_config(bitmap)` method accepts a `bitmap` (a dictionary mapping layer names to bit-widths), and for each layer, it:
    1.  Activates the corresponding LoRA adapter (e.g., `...c_proj-4`).
    2.  Calls `set_bit_width(4)` on the underlying `QuantLinear` layer.

### Step 3 & Step 5: Training Methods and Evaluation

I fine-tune GPT-2 on the generated data using several training strategies, suggested by the papers mentioned in the spec.

-   **`src/trainer.py`**: The `Trainer` class orchestrates the training process. It supports multiple training styles, selectable via the `--training_style` command-line argument.
    -   **`spnet` (Switchable Precision Network)**: In this mode, for each training step, the trainer iterates through a predefined set of quantization bitmaps (e.g., all-8-bit, all-4-bit, a striped 8/4-bit pattern). It aggregates the loss from each configuration, accumulates the gradients, and performs a single optimizer step.
    -   **`cyclic`**: This is Cyclic Precision Training. The trainer cycles through the bit boundary (2-bit to 8-bit) and performs a independent optimizer step for each one.
    -   **`instantnet`**: This is an extension of `spnet` that incorporates Knowledge Distillation. Lower-precision configurations are trained with CELoss and KLDivloss from higher bit-width

### Step 4

I used several scripts to analyze the trained model and find the better trade-off between performance and efficiency.

-   **`quant_model_eval.py`**: The main evaluation script. It loads a trained model and evaluates it on the SQuAD dev set with a quantization configuration specified by the `--bitmap` argument. It reports Exact Match (EM), F1-score, peak VRAM usage, and inference speed (tokens/sec).
-   **`sensitivity_analysis.py`**: This script helps identify which layers are most robust to quantization. It calculates the MSE and MAE between a layer's full-precision output and its quantized output. My intuition is layers with lower MSE are less sensitive and can be quantized more aggressively.
-   **`greedy_search.py`**: This script automates the search for an optimal mixed-precision configuration. Using the sensitivity ranking, it starts with a full-precision model and greedily quantizes layers one by one, from least sensitive to most sensitive. It checks at each step that the drop in F1 score does not exceed a specified budget.
  
## Important Scripts and Usage

### `main.py`

-   **Purpose**: The main entry point for training a switchable quantized model. Will save lora adapters.
-   **Usage**:
    ```bash
    python main.py \
        --training_style [spnet|cyclic|instantnet] \
        --exp_name test \
        --total_steps 1000 \
        --eval_every 100 \
        --use_tensorboard
    ```

### `eval.py`

-   **Purpose**: Evaluate a trained model with a specific quantization bitmap.
-   **Usage**:
    ```bash
    # Evaluate with all layers at 8-bit
    python eval.py --model_path $path_to_lora_adapters --bitmap 8

    # Evaluate with a mixed-precision bitmap
    python eval.py --model_path $path_to_lora_adapters --bitmap "8,8,4,4,8,8,4,4,2,2,2,2,..."
    ```

### `sensitivity_analysis.py`

-   **Purpose**: Rank layers by their sensitivity to quantization to inform manual or automated mixed-precision strategies.
-   **Usage**:
    ```bash
    python sensitivity_analysis.py \
        --model_path $path_to_lora_adapters \
        --output_file sensitivity.json
    ```
    -   This will generate `sensitivity.json`, which stores the MSE for combinations of linear layer and quantization level.

### `greedy_search.py`

-   **Purpose**: Automatically find a high-performing mixed-precision configuration that respects a performance budget.
-   **Usage**:
    ```bash
    python greedy_search.py \
        --model_path $path_to_lora_adapters \
        --sensitivity_file sensitivity.json \
        --budget 0.3 \ # Allows 30% F1 drop
        --output_csv search_results.csv
    ```