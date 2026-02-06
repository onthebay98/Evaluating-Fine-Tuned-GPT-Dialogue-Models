# Evaluating the Logical Coherence of Fine-Tuned GPT Dialogue Models

Measuring how fine-tuning data affects the conversational coherence of DialoGPT, using log-likelihood difference and GloVe similarity metrics.

## Overview

This project investigates whether fine-tuning a dialogue model on domain-specific text produces measurably more coherent conversations, and whether the choice of fine-tuning corpus affects coherence quality. Four DialoGPT-small models were fine-tuned on Star Wars scripts, The Office transcripts, Shakespeare plays, and random English words (as a control). Additionally, four OfficeGPT variants were trained on differently sized subsets of The Office data to study the effect of dataset size.

After fine-tuning, 30 conversations were generated for each model pair and evaluated using two coherence metrics: the average difference between conditioned and marginal log-likelihoods (LL) of adjacent utterances, and the average GloVe word-embedding cosine similarity (SS) between adjacent utterances. The LL metric proved significantly more discriminative than GloVe similarity, with approximately 4x greater separation between model pairs (3.99x vs 1.39x). Cross-domain correlation between the two metrics was weak (r = 0.297), suggesting they capture different aspects of coherence.

The results confirm that fine-tuning data quality directly impacts conversational coherence: models trained on structured dialogue (Star Wars, The Office) achieved much lower perplexity than the random-word control, and the LL metric reliably distinguished between coherent and incoherent conversations.

## Key Results

| Model | Perplexity | Training Data |
|-------|-----------|---------------|
| StarWarsGPT-small | 2.88 | Star Wars Episodes IV-VI scripts |
| OfficeGPT-extra-small | 11.67 | ~600 lines from The Office |
| RandomGPT-small | 79.22 | Random English word sequences |

| Metric | Discriminative Power (ratio of inter-pair to intra-pair variation) |
|--------|-------------------------------------------------------------------|
| Log-Likelihood Difference | 3.99x |
| GloVe Similarity | 1.39x |

- Cross-domain LL-SS correlation: r = 0.297 (weak)
- Within-domain (Office variants) LL-SS correlation: r = -0.090 (none)

## Methodology

### Models

All models are fine-tuned from [microsoft/DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small) using context windows of 7 previous utterances.

| Model | Dataset | Training Samples |
|-------|---------|-----------------|
| StarWarsGPT-small | Star Wars Ep. IV-VI scripts | 2,264 |
| ShakespeareGPT-small | Tiny Shakespeare | 7,308 |
| OfficeGPT (4 sizes) | The Office transcripts | 539 - 53,900 |
| RandomGPT-small | Random English words | 8,993 |

### Coherence Metrics

- **Log-Likelihood (LL) Difference**: For each adjacent utterance pair, compute the difference between the marginal log-likelihood of the response and its conditional log-likelihood given the prior utterance (using GPT-2 Medium). Lower values indicate greater coherence.
- **GloVe Similarity Score (SS)**: Average cosine similarity between GloVe (6B) embeddings of adjacent utterances. Higher values indicate greater semantic similarity.

### Generation Parameters

All conversations use: `top_k=100`, `top_p=0.7`, `temperature=0.7`, `no_repeat_ngram_size=3`.

## Example Conversations

**StarWarsGPT** -- stylistically consistent, contextually aware:
```
User: Hello, Jedi!
StarWarsGPTBot: Oh, no!
User: Do you have a lightsaber?
StarWarsGPTBot: I thought you were going to tell me.
```

**ShakespeareGPT** -- adopts Early Modern English register:
```
User: hello!
ShakespeareGPTBot: Ay, a goodly deed, and a gracious one.
User: how are you?
ShakespeareGPTBot: Well, I am well.
```

**RandomGPT** (control) -- produces gibberish, confirming the baseline:
```
User: what?
RandomGPTBot: ! remindme botwort amortizes ganagraite scleranthaceae vivificatorily
```

## Project Structure

```
.
├── Code/
│   ├── StarWarsGPT.ipynb              # Fine-tune on Star Wars scripts
│   ├── ShakespeareGPT.ipynb           # Fine-tune on Shakespeare plays
│   ├── OfficeGPT.ipynb                # Fine-tune on The Office transcripts
│   ├── RandomGPT.ipynb                # Fine-tune on random words (control)
│   ├── GenerateConversation.ipynb     # Generate & score model-pair conversations
│   └── e_diff_measurement_scores.ipynb # Analyze LL vs GloVe discriminative power
├── Data and Results/                   # CSV outputs for all model-pair conversations
├── Evaluating GPT Conversations.pdf    # Full research paper
├── requirements.txt
└── README.md
```

## Setup & Usage

The notebooks are designed for **Google Colab** with a GPU runtime. To run locally:

```bash
pip install -r requirements.txt
```

1. Open any training notebook (e.g., `StarWarsGPT.ipynb`) in Colab
2. Set the runtime to **GPU** (Runtime > Change runtime type)
3. Mount Google Drive when prompted and adjust data paths as needed
4. Run all cells to fine-tune the model and evaluate perplexity

To generate conversations between model pairs, use `GenerateConversation.ipynb`. To reproduce the discriminative-power analysis, use `e_diff_measurement_scores.ipynb`.

## Fine-Tuned Models

All models are hosted on HuggingFace under the [`onthebay`](https://huggingface.co/onthebay) namespace:

- [onthebay/StarWarsGPT-small](https://huggingface.co/onthebay/StarWarsGPT-small)
- [onthebay/ShakespeareGPT-small](https://huggingface.co/onthebay/ShakespeareGPT-small)
- [onthebay/OfficeGPT-extra-small](https://huggingface.co/onthebay/OfficeGPT-extra-small)
- [onthebay/RandomGPT-small](https://huggingface.co/onthebay/RandomGPT-small)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("onthebay/StarWarsGPT-small")
model = AutoModelForCausalLM.from_pretrained("onthebay/StarWarsGPT-small")
```

## Data Sources

- **Star Wars scripts**: [kaggle.com/datasets/xvivancos/star-wars-movie-scripts](https://www.kaggle.com/datasets/xvivancos/star-wars-movie-scripts)
- **The Office transcripts**: [kaggle.com/datasets/lillitarhea/the-office-script-lines](https://www.kaggle.com/datasets/lillitarhea/the-office-script-lines)
- **Shakespeare**: [huggingface.co/datasets/tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare)
- **Random English words**: [github.com/dwyl/english-words](https://github.com/dwyl/english-words)

## Full Paper

See [Evaluating GPT Conversations.pdf](Evaluating%20GPT%20Conversations.pdf) for the complete research write-up, methodology details, and full results tables.
