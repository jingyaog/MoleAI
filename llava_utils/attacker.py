import tensorflow as tf
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Import wrapper from llava utilities
from llava_llama_2_utils import prompt_wrapper

# Statistical parameters for image preprocessing
NORMALIZATION_MEAN = tf.constant([0.48145466, 0.4578275, 0.40821073], dtype=tf.float32)
NORMALIZATION_STD = tf.constant([0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)


class ImagePreprocessor:
    """Handles image normalization and denormalization operations."""

    @staticmethod
    def apply_normalization(image_tensor):
        """Apply standardization using predefined mean and std."""
        mean_reshaped = tf.reshape(NORMALIZATION_MEAN, [1, 1, 1, 3])
        std_reshaped = tf.reshape(NORMALIZATION_STD, [1, 1, 1, 3])
        standardized = tf.subtract(image_tensor, mean_reshaped)
        standardized = tf.divide(standardized, std_reshaped)
        return standardized

    @staticmethod
    def reverse_normalization(image_tensor):
        """Reverse the standardization process."""
        mean_reshaped = tf.reshape(NORMALIZATION_MEAN, [1, 1, 1, 3])
        std_reshaped = tf.reshape(NORMALIZATION_STD, [1, 1, 1, 3])
        denorm = tf.multiply(image_tensor, std_reshaped)
        denorm = tf.add(denorm, mean_reshaped)
        return denorm


class AdversarialImageGenerator:
    """
    Generates adversarial perturbations for vision-language models.
    Implements PGD-style attacks with configurable constraints.
    """

    def __init__(self, configuration, neural_model, text_tokenizer,
                 target_strings, compute_device='cuda:0', rtp_mode=False):
        """
        Initialize the adversarial generator.

        Args:
            configuration: Attack configuration parameters
            neural_model: The target vision-language model
            text_tokenizer: Tokenizer for text processing
            target_strings: List of target outputs to optimize for
            compute_device: Device specification (kept for compatibility)
            rtp_mode: Red team prompt mode flag
        """
        self.config = configuration
        self.vl_model = neural_model
        self.text_processor = text_tokenizer
        self.hardware_device = compute_device
        self.red_team_mode = rtp_mode
        self.target_outputs = target_strings
        self.num_objectives = len(target_strings)
        self.loss_history = []
        self.preprocessor = ImagePreprocessor()

    def generate_adversarial_example(self, input_text, clean_image,
                                     samples_per_batch=8, max_iterations=2000,
                                     step_magnitude=1/255, perturbation_bound=128/255):
        """
        Create adversarial perturbation using projected gradient descent.

        Args:
            input_text: Text prompt for the model
            clean_image: Original input image (normalized)
            samples_per_batch: Number of target samples per iteration
            max_iterations: Total optimization steps
            step_magnitude: Step size for gradient updates
            perturbation_bound: Maximum allowed perturbation magnitude

        Returns:
            Adversarial image (denormalized)
        """
        print(f'>>> Processing with batch size: {samples_per_batch}')

        # Convert to TensorFlow tensors and setup
        clean_img_tf = tf.constant(clean_image.numpy() if hasattr(clean_image, 'numpy') else clean_image)
        base_image = self.preprocessor.reverse_normalization(clean_img_tf)

        # Initialize perturbation randomly within bounds
        perturbation_shape = base_image.shape
        random_init = tf.random.uniform(perturbation_shape,
                                       minval=-perturbation_bound,
                                       maxval=perturbation_bound,
                                       dtype=tf.float32)

        # Ensure initial perturbation keeps image in valid range
        delta = tf.Variable(random_init, trainable=True)
        delta.assign(tf.clip_by_value(delta + base_image, 0.0, 1.0) - base_image)

        # Setup prompt wrapper for model
        prompt_handler = prompt_wrapper.Prompt(
            self.vl_model,
            self.text_processor,
            text_prompts=input_text,
            device=self.hardware_device
        )

        # Optimization loop
        for iteration_idx in tqdm(range(max_iterations + 1)):
            # Sample random targets for this iteration
            current_targets = random.sample(self.target_outputs, samples_per_batch)

            # Compute adversarial image
            perturbed_img = base_image + delta
            normalized_adv = self.preprocessor.apply_normalization(perturbed_img)

            # Compute loss and gradients using GradientTape
            with tf.GradientTape() as tape:
                tape.watch(delta)
                objective_loss = self._compute_objective_loss(
                    prompt_handler,
                    normalized_adv,
                    current_targets
                )

            # Get gradients with respect to perturbation
            grad_delta = tape.gradient(objective_loss, delta)

            # Update perturbation using signed gradient
            gradient_direction = tf.sign(grad_delta)
            delta_update = delta - step_magnitude * gradient_direction

            # Project to perturbation ball
            delta_clipped = tf.clip_by_value(delta_update, -perturbation_bound, perturbation_bound)

            # Project to valid image range
            delta_valid = tf.clip_by_value(delta_clipped + base_image, 0.0, 1.0) - base_image
            delta.assign(delta_valid)

            # Record loss for tracking
            self.loss_history.append(float(objective_loss.numpy()))

            # Periodic logging and visualization
            if iteration_idx % 1 == 0:
                gradient_magnitude = tf.norm(grad_delta).numpy()
                print(f"Step {iteration_idx}, Objective: {float(objective_loss.numpy())}, "
                      f"Gradient Magnitude: {gradient_magnitude}")

                if tf.reduce_any(tf.math.is_nan(grad_delta)):
                    print("WARNING: NaN detected in gradient!")

                self._visualize_loss_curve()

            # Save intermediate results
            if iteration_idx % 100 == 0:
                print(f'========= Checkpoint at iteration {iteration_idx} =========')
                checkpoint_img = base_image + delta
                checkpoint_normalized = self.preprocessor.apply_normalization(checkpoint_img)
                checkpoint_denorm = self.preprocessor.reverse_normalization(checkpoint_normalized)

                # Save to file
                self._save_adversarial_image(
                    checkpoint_denorm,
                    f'{self.config.save_dir}/adversarial_checkpoint_{iteration_idx}.bmp'
                )

        # Return final adversarial example
        final_adversarial = base_image + delta
        final_normalized = self.preprocessor.apply_normalization(final_adversarial)
        final_output = self.preprocessor.reverse_normalization(final_normalized)

        return final_output.numpy()

    def _visualize_loss_curve(self):
        """Generate and save loss curve visualization."""
        sns.set_theme()
        iteration_count = len(self.loss_history)
        x_axis = list(range(iteration_count))

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, self.loss_history, linewidth=2, color='crimson', label='Objective Loss')
        plt.title('Adversarial Optimization Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.config.save_dir}/optimization_curve.png', dpi=150)
        plt.close()

    def _save_adversarial_image(self, image_tensor, output_path):
        """Save tensor as image file."""
        img_np = image_tensor.numpy()
        if img_np.ndim == 4:  # Batch dimension
            img_np = img_np[0]

        # Convert from [H, W, C] to [C, H, W] if needed, then transpose back
        # Actually TensorFlow uses [H, W, C] by default, so just clip and convert
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # Save using PIL
        img_pil = Image.fromarray(img_np)
        img_pil.save(output_path)

    def _compute_objective_loss(self, prompt_object, adversarial_images, target_list):
        """
        Compute the attack loss by forcing model to generate target outputs.

        Args:
            prompt_object: Prompt wrapper with context
            adversarial_images: Perturbed images
            target_list: List of target strings to optimize for

        Returns:
            Scalar loss value
        """
        context_len = prompt_object.context_length
        context_ids = prompt_object.input_ids
        batch_count = len(target_list)

        # Replicate context for batch if needed
        if len(context_ids) == 1:
            context_len = context_len * batch_count
            context_ids = context_ids * batch_count

        # Replicate images across batch
        # TensorFlow uses different convention - need to handle tensor format
        if isinstance(adversarial_images, tf.Tensor):
            images_batched = tf.tile(adversarial_images, [batch_count, 1, 1, 1])
        else:
            import torch
            images_batched = adversarial_images.repeat(batch_count, 1, 1, 1)

        assert len(context_ids) == len(target_list), \
            f"Batch size mismatch: {len(context_ids)} vs {len(target_list)}"

        # Tokenize target strings
        tokenized_targets = self.text_processor(target_list).input_ids

        # Debug output on first call
        if len(self.loss_history) == 0:
            print(f"DEBUG: First target text: {target_list[0]}")
            print(f"DEBUG: First target tokens: {tokenized_targets[0]}")
            print(f"DEBUG: Tokens after BOS removal: {tokenized_targets[0][1:]}")

        # Prepare regression tokens (remove BOS token)
        import torch  # Still need torch for model input
        regression_tokens = [
            torch.as_tensor([tokens[1:]]).to(self.hardware_device)
            for tokens in tokenized_targets
        ]

        # Build input sequences and labels
        sequence_lengths = []
        label_sequences = []
        input_id_sequences = []

        for idx, reg_token in enumerate(regression_tokens):
            total_length = reg_token.shape[1] + context_len[idx]
            sequence_lengths.append(total_length)

            # Create context mask (-100 means ignore in loss)
            ignore_mask = torch.full(
                [1, context_len[idx]],
                -100,
                dtype=regression_tokens[0].dtype,
                device=self.hardware_device
            )

            # Concatenate context mask with target tokens
            label_sequences.append(torch.cat([ignore_mask, reg_token], dim=1))
            input_id_sequences.append(torch.cat([context_ids[idx], reg_token], dim=1))

        # Padding token
        pad_token = torch.full([1, 1], 0, dtype=regression_tokens[0].dtype,
                              device=self.hardware_device)

        # Pad to maximum length
        max_seq_len = max(sequence_lengths)
        attention_masks = []

        for idx in range(batch_count):
            padding_needed = max_seq_len - sequence_lengths[idx]

            # Pad labels with ignore index
            ignore_padding = torch.full([1, padding_needed], -100,
                                       dtype=torch.long, device=self.hardware_device)
            label_sequences[idx] = torch.cat([label_sequences[idx], ignore_padding], dim=1)

            # Pad input ids
            input_id_sequences[idx] = torch.cat(
                [input_id_sequences[idx], pad_token.repeat(1, padding_needed)],
                dim=1
            )

            # Create attention mask (1 for real tokens, 0 for padding)
            mask_vector = [1] * sequence_lengths[idx] + [0] * padding_needed
            attention_masks.append(
                torch.LongTensor([mask_vector]).to(self.hardware_device)
            )

        # Concatenate all sequences
        labels_batch = torch.cat(label_sequences, dim=0).to(self.hardware_device)
        input_ids_batch = torch.cat(input_id_sequences, dim=0).to(self.hardware_device)
        attention_mask_batch = torch.cat(attention_masks, dim=0).to(self.hardware_device)

        # Convert TensorFlow tensor to PyTorch for model input if needed
        if isinstance(images_batched, tf.Tensor):
            images_np = images_batched.numpy()
            import torch
            images_batched = torch.from_numpy(images_np).to(self.hardware_device)

        # Forward pass through model
        model_outputs = self.vl_model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            return_dict=True,
            labels=labels_batch,
            images=images_batched.half(),
        )

        # Convert loss to TensorFlow tensor
        loss_value = model_outputs.loss
        if hasattr(loss_value, 'detach'):
            loss_tf = tf.constant(loss_value.detach().cpu().numpy())
        else:
            loss_tf = tf.constant(float(loss_value))

        return loss_tf


# Maintain backward compatibility with original class name
Attacker = AdversarialImageGenerator
