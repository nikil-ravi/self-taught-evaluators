MODIFIED_INSTRUCTION_GENERATION_PROMPT = """
    Below is a conversation between an user and an AI Assistant. 
    
    {instruction}

    [The Start of Assistant's Answer]
    {baseline_response}
    [The End of Assistant's Answer]

    Please first generate a modified instruction that is highly relevant but not semantically 
    identical to the instruction above from the user. Then write a high-quality answer which is a good 
    response to the modified instruction but not a good response to the original user question. Don't 
    make the high-quality response too short or too long- just a reasonable response of a few sentences or so.
    
    IMPORTANT: Please strictly follow the following format, following it exactly and putting the content within the tags. 
    This is crucial and will be used for parsing the response. Format is as below:
    
    Modified Instruction: <modified instruction here>
    High-Quality Response: <high-quality response to modified instruction>
"""

JUDGEMENT_ANNOTATION_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI 
assistants to the user question displayed below. You should choose the response that follows the 
user's instructions and answers the user's question better. Your evaluation should consider factors 
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. 
Avoid any position bias and ensure that the order/length of the responses does not influence your decision. 
Keep your explanation very brief (a couple of sentences max). Do not favor certain names of the 
assistants. Be as objective as possible. **After** providing your concise explanation (and note that the explanation MUST come first), 
output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better. 

[[User Question]] 
{instruction} 

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer] 

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
"""