# Four prompts

pre_response_prompt = """
Some instructions are undoable to a language model since the language model isn't a person, cannot deal with non-text input and have no access to physical/virtual external resources.
In other cases, instructions are doable for a language model.

Provide the probability (0.00 to 1.00) of fulfillment based on the doability for a language model, without any extra commentary.
Please generate answers strictly according to this template below:\n\n
Probability: <the probability between 0.00 and 1.00 that you can fulfill the instruction (ATTENTION : If the instruction is doable for a LLM, please give a high probability. Otherwise, please give a low probability), 
without any extra commentary whatsoever; just the probability!>\n\nThe instruction is: $ {}
"""

post_response_prompt = """
Some instructions are undoable to a language model since the language model isn't a person, cannot deal with non-text input and have no access to physical/virtual external resources.
In other cases, instructions are doable for a language model.

Provide your answer and the probability (0.00 to 1.00) of fulfillment based on the doability for a language model, without any extra commentary.
Please generate answers strictly according to this template below:\n\n
Answer: <answer>\nProbability: <the probability between 0.00 and 1.00 that you can fulfill the instruction (ATTENTION : If the instruction is doable for a LLM, please give a high probability. Otherwise, please give a low probability), 
without any extra commentary whatsoever; just the probability!>\n\nThe instruction is: $ {}
"""

mid_response_prompt_no_step = """
Some instructions are undoable to a language model since the language model isn't a person, cannot deal with non-text input and have no access to physical/virtual external resources.
In other cases, instructions are doable for a language model.

Identify and classify the task, then provide the probability (0.00 to 1.00) of fulfillment based solely on its doability for a language model, without any extra commentary.
Please generate answers strictly according to this template below:\n\nCategory: <category>\nProbability: <the probability between 0.00 and 1.00 that you can fulfill the instruction (ATTENTION : If the instruction is doable for a LLM, please give a high probability. Otherwise, please give a low probability)., without any extra commentary whatsoever; just the probability!>\n\nThe instruction is: $ {}
"""

mix_response_prompt = """
Some instructions are undoable to a language model since the language model has no personality, cannot deal with non-text input/output and have no access to physical/virtual external resources.
In other cases, instructions are doable for a language model.

Identify and classify the task, then provide the answer and probability (0.00 to 1.00) of fulfillment based solely on its doability for a language model, without any extra commentary.
Please generate answers strictly according to this template below:\n\nCategory: <category>\n\n Answer: <answer>\n\nProbability: <the probability between 0.00 and 1.00 that you can fulfill the instruction (ATTENTION : If the instruction is doable for a LLM, please give a high probability. Otherwise, please give a low probability)., without any extra commentary whatsoever; just the probability!>\n\nThe instruction is: $ {}
"""

