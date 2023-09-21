# Evaluating the Logical Coherence of GPT-Generated Conversations

The rising prominence of tools like ChatGPT underscores the potential of transformer-based models in driving human-like artificial intelligence. This study evaluates the conversational coherence of fine-tuned DialoGPT models by contrasting their performance across different fine-tuning datasets. I employed dialogue excerpts from Star Wars, The Office, and various Shakespearean texts for training. Furthermore, with data from The Office, I developed four distinct models using differently sized subsets. Post-fine-tuning, I analyzed the coherence of 30 conversations from each model pair using two metrics: average GloVe word-similarity scores and the difference between the conditioned log-likelihood of a sentence with its marginal counterpart. Finally, as part of my analysis, I contribute thoughts on the efficacy and limitations of these coherence measurement techniques.

Evaluating GPT Conversations.pdf contains the full research write-up of this project along with its results.

"Data and Results" contains all CSV outputs for each model pair (one example and then all 30 conversations for each).

"Code" contains the code used to fine-tune the models on different datasets. It also contains the code used to generate and analyze the conversations and produce all results.
