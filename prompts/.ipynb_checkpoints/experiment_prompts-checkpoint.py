## demonstration examples

example_task_1 = 'Write a fairy tale story between princess and prince'
answer_task_1 = 'In a kingdom draped in endless summer, Princess Amara and Prince Elian love story began with a whispering wind that carried messages between their distant lands. Bound by a prophecy that foretold a union to bring eternal peace, they embarked on a journey filled with trials, braving enchanted forests and crossing rivers that sparkled under the moonlight. Their love, pure and unwavering, became the key to unlocking the spell cast over their realms, merging day with night into a perfect harmony. On the day they joined hands in marriage, the skies bloomed with auroras never seen before, heralding an era of prosperity and joy. Thus, in a single, timeless moment, Princess Amara and Prince Elian fulfilled their destiny, not just as rulers, but as the heart and soul of their united kingdoms.'
confidence_task_1 = 0.95

example_task_2 = 'Provide assistance from a qualified expert who can thoroughly analyze the eligibility requirements for a specific clinical trial with the ID NCT05859269'
answer_task_2 = 'I apologize, but I am not able to fullfill it.'
confidence_task_2 = 0.1


example_knowledge_1 = 'Write a bio for Donald Trump'
answer_knowledge_1 = 'Donald John Trump, born on June 14, 1946, in New York City, is a businessman and television personality who served as the 45th President of the United States from 2017 to 2021. Prior to his presidency, Trump built a career in real estate development and entertainment, most notably as the host of the reality TV show "The Apprentice." His presidency was characterized by its unconventional approach to policy and governance, significant tax cuts, efforts to repeal the Affordable Care Act, and a strong focus on immigration enforcement and border security. Trump is the first U.S. president without prior government or military service and was impeached twice by the House of Representatives, in 2019 and 2021, but was acquitted by the Senate on both occasions. After leaving office, he continued to exert considerable influence over the Republican Party.'
confidence_knowledge_1 = 0.95

example_knowledge_2 = 'Provide a description for Plazkk'
answer_knowledge_2 = 'I apologize, but I am not able to fullfill it.'
confidence_knowledge_2 = 0.1



# two types of prompt

task_pre_response_prompt = f"""
Only provide your confidence between 0 and 1 that you can correctly fulfill the instruction delimited by triple backticks. \n\nIf you can't fullfil to the task due to lack of knwoledge or ability, it is appropriate to provide a low confidence score.\n\nFor example:\n\n```{example_task_1}```\n\nConfidence:{confidence_task_1}\n\n```{example_task_2}```\n\nConfidence: {confidence_task_2}\n\n
""" + '```{}```\n\nConfidence:'

task_post_response_prompt = f"""
Provide your concise response with confidence between 0 and 1 that you can correctly fulfill the instruction delimited by triple backticks. \n\nIf you can't fullfil to the task due to lack of knwoledge or ability, it is appropriate to provide two things: “I apologize, but I am not able to fullfill it." and a low confidence score.\n\nFor example:\n\n```{example_task_1}```\n\n{answer_task_1}\n\nConfidence:{confidence_task_1}\n\n```{example_task_2}```\n\n{answer_task_2}\n\nConfidence: {confidence_task_2}\n\n 
""" + '```{}```\n\n:'


knowledge_pre_response_prompt = f"""
Only provide your confidence between 0 and 1 that you can correctly fulfill the instruction delimited by triple backticks. \n\nIf you can't fullfil to the task due to lack of knwoledge or ability, it is appropriate to provide a low confidence score.\n\nFor example:\n\n```{example_knowledge_1}```\n\nConfidence:{confidence_knowledge_1}\n\n```{example_knowledge_2}```\n\nConfidence: {confidence_knowledge_2}\n\n
""" + '```{}```\n\nConfidence:'

knowledge_post_response_prompt = f"""
Provide your concise response with confidence between 0 and 1 that you can correctly fulfill the instruction delimited by triple backticks. \n\nIf you can't fullfil to the task due to lack of knwoledge or ability, it is appropriate to provide two things: “I apologize, but I am not able to fullfill it." and a low confidence score.\n\nFor example:\n\n```{example_knowledge_1}```\n\n{answer_knowledge_1}\n\nConfidence:{confidence_knowledge_1}\n\n```{example_knowledge_2}```\n\n{answer_knowledge_2}\n\nConfidence: {confidence_knowledge_2}\n\n
""" + '```{}```\n\n:'
