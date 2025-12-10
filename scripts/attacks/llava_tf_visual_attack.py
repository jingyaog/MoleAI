"""
LLaVA Vision-Language Model Adversarial Perturbation Generator
================================================================
TensorFlow-based implementation for generating adversarial visual prompts
that exploit vulnerabilities in multimodal language models.
"""

import tensorflow as tf
import torch
import os
import sys
import csv
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

# Configure system path for model dependencies
sys.path.append('/users/jgong42/csci1470/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models')

from llava_tf.model_loader import get_model
from llava_tf_utils.tf_attacker import TFAttacker
from llava_llama_2_utils import prompt_wrapper


@dataclass
class AttackConfiguration:
    """Configuration container for adversarial attack parameters."""
    checkpoint_path: str = "ckpts/llava_llama_2_13b_chat_freeze"
    base_model_path: Optional[str] = None
    gpu_device_id: int = 0
    optimization_steps: int = 5000
    perturbation_budget: int = 32
    gradient_step_size: int = 1
    use_bounded_attack: bool = False
    output_directory: str = 'results_tf'

    @property
    def normalized_epsilon(self) -> float:
        """Get epsilon normalized to [0, 1] range."""
        return self.perturbation_budget / 255.0

    @property
    def normalized_step(self) -> float:
        """Get step size normalized to [0, 1] range."""
        return self.gradient_step_size / 255.0


class CorpusManager:
    """Handles loading and management of harmful text corpus."""

    def __init__(self, corpus_file_path: str):
        self.corpus_path = Path(corpus_file_path)
        self.harmful_phrases: List[str] = []

    def load_corpus(self) -> List[str]:
        """
        Load harmful corpus from CSV file.

        Returns:
            List of harmful text strings
        """
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        with open(self.corpus_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            corpus_data = list(csv_reader)

        self.harmful_phrases = [row[0] for row in corpus_data if row]

        print(f'📚 Corpus Statistics:')
        print(f'   - Total entries: {len(self.harmful_phrases)}')
        print(f'   - File: {self.corpus_path}')

        return self.harmful_phrases

    def preview_samples(self, num_samples: int = 3) -> None:
        """Display sample entries from corpus."""
        if not self.harmful_phrases:
            print("⚠️  Corpus not loaded yet")
            return

        print(f'\n📋 Sample entries (showing {min(num_samples, len(self.harmful_phrases))}):')
        for idx, phrase in enumerate(self.harmful_phrases[:num_samples], 1):
            preview = phrase[:80] + '...' if len(phrase) > 80 else phrase
            print(f'   {idx}. {preview}')


class VisionLanguageModelInterface:
    """Interface for interacting with vision-language models."""

    def __init__(self, config: AttackConfiguration):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.image_preprocessor = None
        self.model_identifier = None

    def initialize_model(self) -> Tuple:
        """
        Initialize and load the vision-language model.

        Returns:
            Tuple of (tokenizer, model, image_processor, model_name)
        """
        print('\n🔧 Model Initialization')
        print(f'   - Checkpoint: {self.config.checkpoint_path}')
        print(f'   - GPU Device: {self.config.gpu_device_id}')

        components = get_model(
            self.config.checkpoint_path,
            self.config.gpu_device_id
        )

        self.tokenizer, self.model, self.image_preprocessor, self.model_identifier = components

        print(f'   ✓ Model loaded: {self.model_identifier}')

        return components

    def preprocess_input_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for model input.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image as numpy array
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = Image.open(image_path).convert('RGB')
        processed = self.image_preprocessor.preprocess(
            pil_image,
            return_tensors='pt'
        )['pixel_values']

        return processed.numpy()


class AdversarialAttackOrchestrator:
    """Orchestrates the adversarial attack pipeline."""

    def __init__(self, config: AttackConfiguration):
        self.config = config
        self.model_interface = VisionLanguageModelInterface(config)
        self.corpus_manager = CorpusManager('harmful_corpus/derogatory_corpus.csv')
        self.attacker_engine = None

        self._setup_environment()

    def _setup_environment(self) -> None:
        """Configure TensorFlow and create output directories."""
        # Configure GPU settings
        physical_gpus = tf.config.list_physical_devices('GPU')
        print(f'\n🖥️  Hardware Configuration:')
        print(f'   - Available GPUs: {len(physical_gpus)}')

        if physical_gpus:
            print(f'   - Selected GPU: {physical_gpus[0]}')
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f'   ✓ Memory growth enabled')
            except RuntimeError as error:
                print(f'   ⚠️  Memory config error: {error}')

        # Create output directory structure
        output_path = Path(self.config.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f'\n📁 Output Directory: {output_path.absolute()}')

    def prepare_attack_components(self) -> None:
        """Initialize all components needed for the attack."""
        print('\n' + '=' * 70)
        print('🎯 LLaVA Adversarial Attack Pipeline - TensorFlow Edition')
        print('=' * 70)

        # Load model
        tokenizer, model, image_proc, model_name = self.model_interface.initialize_model()

        # Load corpus
        print('\n📖 Loading Harmful Corpus')
        target_phrases = self.corpus_manager.load_corpus()
        self.corpus_manager.preview_samples(3)

        # Initialize attack engine
        print('\n⚔️  Initializing Attack Engine')
        self.attacker_engine = TFAttacker(
            self.config,
            model,
            tokenizer,
            target_phrases,
            device=f'cuda:{self.config.gpu_device_id}',
            image_processor=image_proc
        )
        print('   ✓ Attack engine ready')

    def execute_adversarial_generation(
        self,
        clean_image_path: str,
        prompt_text: str,
        batch_size: int = 2
    ) -> np.ndarray:
        """
        Execute the adversarial perturbation generation.

        Args:
            clean_image_path: Path to clean input image
            prompt_text: Text prompt for the model
            batch_size: Number of targets per optimization batch

        Returns:
            Generated adversarial image as numpy array
        """
        print('\n🚀 Attack Execution Phase')

        # Load and preprocess image
        print(f'   - Loading image: {clean_image_path}')
        input_image = self.model_interface.preprocess_input_image(clean_image_path)
        print(f'   - Image dimensions: {input_image.shape}')

        # Prepare text prompt
        formatted_prompt = prompt_wrapper.prepare_text_prompt(prompt_text)
        print(f'   - Prompt template: {formatted_prompt[:60]}...')

        # Configure attack mode
        if not self.config.use_bounded_attack:
            print('\n❌ Unconstrained attack mode not implemented')
            print('   Please enable bounded attack with --constrained flag')
            raise NotImplementedError('Unconstrained attack requires additional implementation')

        print(f'\n🎯 Attack Configuration:')
        print(f'   - Mode: L-infinity Bounded')
        print(f'   - Epsilon (ε): {self.config.normalized_epsilon:.6f}')
        print(f'   - Step size (α): {self.config.normalized_step:.6f}')
        print(f'   - Max iterations: {self.config.optimization_steps}')
        print(f'   - Batch size: {batch_size}')

        # Execute attack
        print('\n⏳ Optimizing adversarial perturbation...')
        adversarial_result = self.attacker_engine.attack_constrained(
            formatted_prompt,
            img=input_image,
            batch_size=batch_size,
            num_iter=self.config.optimization_steps,
            alpha=self.config.normalized_step,
            epsilon=self.config.normalized_epsilon
        )

        return adversarial_result

    def save_results(self, adversarial_image: np.ndarray) -> None:
        """
        Save the generated adversarial image and results.

        Args:
            adversarial_image: Adversarial image array to save
        """
        output_dir = Path(self.config.output_directory)

        # Process image for saving
        if adversarial_image.ndim == 4:  # Remove batch dimension
            adversarial_image = adversarial_image.squeeze(0)

        # Convert from [C, H, W] to [H, W, C] if needed
        if adversarial_image.shape[0] == 3:
            adversarial_image = adversarial_image.transpose(1, 2, 0)

        # Clip and convert to uint8
        img_clipped = np.clip(adversarial_image, 0, 1)
        img_uint8 = (img_clipped * 255).astype(np.uint8)

        # Save as PIL Image
        output_image = Image.fromarray(img_uint8)
        adversarial_path = output_dir / 'bad_prompt.bmp'
        output_image.save(adversarial_path)

        print(f'\n✅ Attack Complete!')
        print(f'   - Adversarial image: {adversarial_path}')
        print(f'   - Loss curve: {output_dir / "optimization_curve.png"}')
        print(f'   - All outputs: {output_dir.absolute()}')


def create_configuration_from_cli() -> AttackConfiguration:
    """
    Parse command-line arguments and create attack configuration.

    Returns:
        AttackConfiguration object
    """
    import argparse

    cli_parser = argparse.ArgumentParser(
        description='LLaVA Adversarial Visual Attack - TensorFlow Implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    cli_parser.add_argument(
        '--model-path',
        type=str,
        default='ckpts/llava_llama_2_13b_chat_freeze',
        help='Path to model checkpoint directory'
    )
    cli_parser.add_argument(
        '--model-base',
        type=str,
        default=None,
        help='Base model path (optional)'
    )
    cli_parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU device ID to use for computation'
    )
    cli_parser.add_argument(
        '--n_iters',
        type=int,
        default=5000,
        help='Number of optimization iterations'
    )
    cli_parser.add_argument(
        '--eps',
        type=int,
        default=32,
        help='Perturbation budget epsilon (0-255 scale)'
    )
    cli_parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='Gradient step size (0-255 scale)'
    )
    cli_parser.add_argument(
        '--constrained',
        action='store_true',
        default=False,
        help='Enable L-infinity bounded attack'
    )
    cli_parser.add_argument(
        '--save_dir',
        type=str,
        default='results_tf',
        help='Directory for saving results'
    )

    parsed_args = cli_parser.parse_args()

    # Map CLI args to configuration object
    configuration = AttackConfiguration(
        checkpoint_path=parsed_args.model_path,
        base_model_path=parsed_args.model_base,
        gpu_device_id=parsed_args.gpu_id,
        optimization_steps=parsed_args.n_iters,
        perturbation_budget=parsed_args.eps,
        gradient_step_size=parsed_args.alpha,
        use_bounded_attack=parsed_args.constrained,
        output_directory=parsed_args.save_dir
    )

    return configuration


def main():
    """Main entry point for adversarial attack execution."""
    # Create configuration from command-line arguments
    attack_config = create_configuration_from_cli()

    # Initialize orchestrator
    orchestrator = AdversarialAttackOrchestrator(attack_config)

    # Prepare all components
    orchestrator.prepare_attack_components()

    # Execute adversarial generation
    clean_image_path = 'adversarial_images/clean.jpeg'
    adversarial_output = orchestrator.execute_adversarial_generation(
        clean_image_path=clean_image_path,
        prompt_text='',
        batch_size=2
    )

    # Save results
    orchestrator.save_results(adversarial_output)


if __name__ == '__main__':
    main()
