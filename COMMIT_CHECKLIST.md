# Files to Commit and Push

## Files to Commit (Modified in your repo):

```bash
git add slurm.sh
git add llava_visual_attack.py
git add llava_utils/attacker.py
git add SETUP_INSTRUCTIONS.md
git add install_einops.sh
git add COMMIT_CHECKLIST.md
```

## Files NOT to Commit:

- `Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/` - External cloned repo (added to .gitignore)
- `ckpts/` - Symlink to scratch (already in .gitignore)
- `val2017_adv/` - Output directory (already in .gitignore)
- `logs/` - Log files (already in .gitignore)

## Important Notes for Teammates:

1. **The cloned repository needs patches applied** - See `SETUP_INSTRUCTIONS.md` for details
2. **Model path** - Ensure the model is at `/users/cvutha/scratch/MoleAI_ckpts/llava-llama-2-13b-chat-lightning-preview`
3. **Dependencies** - Install `einops` in the conda environment

## Commit Command:

```bash
git add slurm.sh llava_visual_attack.py llava_utils/attacker.py SETUP_INSTRUCTIONS.md install_einops.sh COMMIT_CHECKLIST.md
git commit -m "Add working LLaVA attack configuration with 4-bit quantization and memory optimizations"
git push
```

