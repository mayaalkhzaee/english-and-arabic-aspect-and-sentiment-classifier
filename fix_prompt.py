import os
import json
from time import sleep
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ===========================
# Load API Key
# ===========================
load_dotenv(".env")

llm = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    model="qwen-plus-2025-04-28",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    max_retries=3,
    timeout=None
)

# ===========================
# Config
# ===========================
INPUT_FILE = "sentences.txt"
OUTPUT_FILE = "aspect_results.jsonl"
BATCH_SIZE = 10


# ===========================
# Universal ABSA Prompt
# ===========================
INSTRUCTIONS = """
Can you help me improve the following LLM prompt, I am still getting failing test cases:
You are performing Aspect-Based Sentiment Analysis (ABSA).

Your task is to extract ONLY explicit aspect terms and their sentiment polarity 
from each sentence. Follow these strict rules:

1. Extract an aspect term ONLY if it is explicitly being evaluated with sentiment 
   in the sentence. 
   - A noun or noun phrase that is merely mentioned is NOT an aspect.
   - There must be a clear opinion directed at that noun phrase.

2. Valid aspect terms are concrete features, components, attributes, or entities 
   that receive direct sentiment.
   Examples: battery life, keyboard, GUI, screen, warranty, service center.

3. DO NOT extract:
   - vague nouns (e.g., "problem", "issue", "reason", "thing", "stuff")
   - pronouns (e.g., "this one", "it", "mine", "my mac")
   - product names unless the entire product is clearly evaluated with sentiment
   - nouns not being evaluated (e.g., “laptop” in “I purchased this laptop”)
   - numbers, quantities, or ranges (e.g., “hundreds or thousands”)
   - mentions of parts that are not being evaluated (e.g., “broken part” if the sentiment is about the company process)

4. Use the exact literal minimal noun phrase as written in the sentence.
   - Do NOT shorten or expand the phrase.
   - Example: “customer service number” must NOT be changed to “customer service”.

5. Extract ALL explicit aspect terms in a sentence. 
   - A sentence may contain multiple aspects.

6. If a sentence expresses sentiment but does NOT explicitly target a feature, 
   return an empty list.

7. Polarity must be one of:
   - positive
   - negative
   - neutral

Return the output in EXACTLY this JSON format:

[
  {
    "id": <sentence_id>,
    "sentence": "<text>",
    "aspect_terms": [
      {"term": "...", "polarity": "..."},
      {"term": "...", "polarity": "..."}
    ]
  }
]

If a sentence has NO explicit aspect terms:

"aspect_terms": []

These are some failing test cases:
Sentence #1: I charge it at night and skip taking the cord with me because of the good battery life.
  GOLD: [('battery life', 'positive'), ('cord', 'neutral')]
  PRED: [('battery life', 'positive')]
  Missing aspects: [('cord', 'neutral')]
  Extra aspects:   []
---------------------------------------------------------
Sentence #2: I bought a HP Pavilion DV4-1222nr laptop and have had so many problems with the computer.
  GOLD: []
  PRED: [('computer', 'negative')]
  Missing aspects: []
  Extra aspects:   [('computer', 'negative')]
---------------------------------------------------------
Sentence #3: The tech guy then said the service center does not do 1-to-1 exchange and I have to direct my concern to the "sales" team, which is the retail shop which I bought my netbook from.
  GOLD: [('service center', 'negative'), ('"sales" team', 'negative'), ('tech guy', 'neutral')]
  PRED: []
  Missing aspects: [('service center', 'negative'), ('"sales" team', 'negative'), ('tech guy', 'neutral')]
  Extra aspects:   []
---------------------------------------------------------
Sentence #6: it is of high quality, has a killer GUI, is extremely stable, is highly expandable, is bundled with lots of very good applications, is easy to use, and is absolutely gorgeous.
  GOLD: [('applications', 'positive'), ('quality', 'positive'), ('gui', 'positive'), ('use', 'positive')]
  PRED: [('applications', 'positive'), ('quality', 'positive'), ('gui', 'positive')]
  Missing aspects: [('use', 'positive')]
  Extra aspects:   []
---------------------------------------------------------
Sentence #7: Easy to start up and does not overheat as much as other laptops.
  GOLD: [('start up', 'positive')]
  PRED: [('start up', 'positive'), ('laptops', 'negative')]
  Missing aspects: []
  Extra aspects:   [('laptops', 'negative')]
---------------------------------------------------------
Sentence #9: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!  
  GOLD: [('features', 'positive'), ('ichat', 'positive'), ('garage band', 'positive'), ('photobooth', 'positive')]
  PRED: [('features', 'positive')]
  Missing aspects: [('ichat', 'positive'), ('garage band', 'positive'), ('photobooth', 'positive')]
  Extra aspects:   []
---------------------------------------------------------
Sentence #10: Needless to say a PC that can't support a cell phone is less than useless!
  GOLD: []
  PRED: [('pc', 'negative')]
  Missing aspects: []
  Extra aspects:   [('pc', 'negative')]
---------------------------------------------------------
Sentence #11: Great laptop that offers many great features!
  GOLD: [('features', 'positive')]
  PRED: [('features', 'positive'), ('laptop', 'positive')]
  Missing aspects: []
  Extra aspects:   [('laptop', 'positive')]
---------------------------------------------------------
Sentence #12: they have done absolutely nothing to fix the computer problem.
  GOLD: []
  PRED: [('computer problem', 'negative')]
  Missing aspects: []
  Extra aspects:   [('computer problem', 'negative')]
---------------------------------------------------------
Sentence #13: One night I turned the freaking thing off after using it, the next day I turn it on, no GUI, screen all dark, power light steady, hard drive light steady and not flashing as it usually does.
  GOLD: [('screen', 'negative'), ('gui', 'negative'), ('hard drive light', 'negative'), ('power light', 'neutral')]
  PRED: [('screen', 'negative'), ('gui', 'negative')]
  Missing aspects: [('hard drive light', 'negative'), ('power light', 'neutral')]
  Extra aspects:   []
---------------------------------------------------------
Sentence #14: Still pretty pricey, but I've been putting off money for a while as a little Macbook Fund, and finally got to use it. 
  GOLD: []
  PRED: [('pricey', 'negative')]
  Missing aspects: []
  Extra aspects:   [('pricey', 'negative')]
---------------------------------------------------------
Sentence #15: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
  GOLD: [('battery', 'neutral')]
  PRED: [('blue screen', 'negative')]
  Missing aspects: [('battery', 'neutral')]
  Extra aspects:   [('blue screen', 'negative')]
---------------------------------------------------------
Sentence #16: In the shop, these MacBooks are encased in a soft rubber enclosure - so you will never know about the razor edge until you buy it, get it home, break the seal and use it (very clever con).
  GOLD: [('rubber enclosure', 'positive'), ('edge', 'negative')]
  PRED: [('razor edge', 'negative')]
  Missing aspects: [('rubber enclosure', 'positive'), ('edge', 'negative')]
  Extra aspects:   [('razor edge', 'negative')]
---------------------------------------------------------
Sentence #17: However, the multi-touch gestures and large tracking area make having an external mouse unnecessary (unless you're gaming).
  GOLD: [('multi-touch gestures', 'positive'), ('tracking area', 'positive'), ('external mouse', 'neutral'), ('gaming', 'neutral')] 
  PRED: [('multi-touch gestures', 'positive'), ('tracking area', 'positive')]
  Missing aspects: [('external mouse', 'neutral'), ('gaming', 'neutral')]
  Extra aspects:   []
---------------------------------------------------------
Sentence #18: Plus it is small and reasonably light so I can take it with me to and from work.
  GOLD: []
  PRED: [('small', 'positive'), ('light', 'positive')]
  Missing aspects: []
  Extra aspects:   [('small', 'positive'), ('light', 'positive')]
---------------------------------------------------------

"""

response = llm.invoke(INSTRUCTIONS)
print(response.content)