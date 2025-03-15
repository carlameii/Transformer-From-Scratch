# Simple transformer implementation from scratch
## Authors:
 - Brendan McKinley
 - Bingshen Lu
 - Carla Ellefsen
 - Diya Vinod
 - Michael Ivanitskiy

## Overview
This is an implementation of a simple transformer for the Spring 2025 course: SP TPS: Large Language Model MATH598BA taught by Dr. Samy Wu Fung and Michael Ivanitskiy at the Colorado School of Mines.

## Author Contributions
All four of us closely collaborated to implement the majority of the Transformer architecture itself, particularly the `AttentionHead`, `TransformerBlock`, and `Transformer` classes. Contributions that were more individual efforts are described below by collaborator:
- Brendan McKinley: Worked on implementing the byte-pair encoding tokenization, multilayer perceptrons, Trainer class, and miscellaneous debugging tasks. 
- Bingshen Lu:
- Carla Ellefsen: Worked on implementing the multiheaded attention mechanism, positional encodings, loading the dataset, and the initial tokenization that was done before the pretrained Tokenizer was used instead. Additionally, was responsible for much of the repository maintenance, such as creating the README, writing issues, merging PRs, etc.
- Diya Vinod: 

## Language Model Use
Generative AI was partially used to help implement this simple language model. ChatGPT was used to help implement the multi-head attention mechanism and positional encodings, and GitHub CoPilot was also useful for connecting the pieces of the Transformer architecture together. 
### Links to ChatGPT Conversations:
 - https://chatgpt.com/share/67d35104-a110-8012-8dc1-41192c5e6041
 - https://chatgpt.com/share/67d3512c-c9fc-8012-880f-2e0b093ebfe5
 - https://chatgpt.com/share/67d35141-47d4-8012-a548-b35d8bc8ec22
 - https://chatgpt.com/share/67d3516c-7720-8012-a562-4b05e746dc4a
 - https://chatgpt.com/share/67d35186-ebf8-8012-8a45-e1feb655afc8
 - https://chatgpt.com/share/67d35193-1fd0-8012-9071-40235b238bfb
 - https://chatgpt.com/share/67d351a1-43e0-8012-afb0-a6d78146119f
