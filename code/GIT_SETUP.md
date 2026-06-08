# Pushing `Robust-NN-learning` to GitHub (`pps121/robustNN`)

Use a **Personal Access Token (PAT)** or **SSH** — never store `HF_TOKEN` or GitHub passwords in the repo (see `.gitignore`).

## One-time setup (on your Mac / local machine)

```bash
cd /path/to/Subho_IIM/Robust-NN-learning

# If this folder is not a git repo yet:
git init
git branch -M main

# Add your remote (HTTPS — you will authenticate with a PAT when pushing)
git remote add origin https://github.com/pps121/robustNN.git
# If `origin` already exists with a wrong URL:
# git remote set-url origin https://github.com/pps121/robustNN.git
```

Create a **GitHub PAT** (classic): Settings → Developer settings → Personal access tokens → repo scope. Use it as the password when `git push` asks (username = your GitHub username).

Or use **SSH** (recommended long-term):

```bash
ssh-keygen -t ed25519 -C "your_email"
# Add ~/.ssh/id_ed25519.pub to GitHub → SSH keys
git remote set-url origin git@github.com:pps121/robustNN.git
```

## Everyday workflow (commit + push)

```bash
cd /path/to/Subho_IIM/Robust-NN-learning
git status
git add part3_BERT_Robust_NLP_Experiments.py part4_Multimodal_Vision_Robust_Experiments.py \
        RunPod_Part3_BERT_Robust_NLP.ipynb RunPod_Part4_Multimodal_Vision.ipynb \
        .gitignore GIT_SETUP.md research_writeup/
git commit -m "Describe your change in one clear sentence."
git push -u origin main
```

If you use a **private branch** instead of `main`, replace branch names accordingly:

```bash
git checkout -b private-research
git push -u origin private-research
```

## Automatic push on every commit?

Git does not auto-push until you run `git push` (or a CI job does). Options:

1. **Shell alias** (local): `alias gp='git push'` after commit.
2. **Git hook**: a `post-commit` hook can run `git push` (easy to mis-push; use with care).
3. **GitHub Actions**: push only on tag/release — overkill for small private repos.

For secret research, **manual `git push` after review** is usually safest.

## RunPod note

Download `runpod_outputs/` (or your `ROBUST_NN_WORKSPACE` folder) **before** stopping the pod — the remote disk is ephemeral.

## Google Colab (running the `.py` directly)

Upload `part3_BERT_Robust_NLP_Experiments.py` or `part4_Multimodal_Vision_Robust_Experiments.py`, then:

```python
!pip install torch transformers datasets ...  # exact list in each script’s docstring
# Set HF_TOKEN via Colab Secrets (lock icon) or the environment panel — do not save secrets in the notebook.
%run part4_Multimodal_Vision_Robust_Experiments.py
```

Or from a cell: `!python part4_Multimodal_Vision_Robust_Experiments.py` after secrets are configured.

## Security

- **Never** commit `HF_TOKEN`, GitHub PATs, or API keys. They belong in RunPod/Colab **environment variables** or **gitignored** `.env` files.
- If a token was ever pasted into a notebook or chat, **revoke it** on the provider’s site and create a new one.
