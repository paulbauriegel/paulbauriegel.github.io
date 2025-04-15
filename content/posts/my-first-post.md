---
title: 'Gemini 2.5 Spatial Reasoning'
date: 2025-04-09T09:23:00+02:00
tags: ['Gemini-2.5', 'Gemini']
---

# Gemini Spatial Reasoning is amazing
... but not good enough to be helpful yet.

Various Large Language Models with vision support enable you to ask questions about images usually refered to as [visual reasoning]. This allows the user to ask questions such as how many pumpkins do you see in this picture or what kind of flower do you see in this picture. This in itself is already really helpful to enable zero-shot classification without training a model. But when you need the position of that pumpkin or that flower in the picture this is not possible. Entering Spatial Resoning, which allows you to get actual bounding boxes or picture coordinates for objects through prompting.

Gemini 2.5 is not the first model to enable this. For the past years most vision models are build on-top of transformers architectures and allow zero-shot prompting to some degree like MobileCLIP, SigLipv2 or GroundingDino, the last even enabiling gounding box not just pure classification. In addition to that it was possible with other vision supporting LLM like GPT4o to apply something called visual grounding, but that was complicated to build and not really worth it from my perspective compared to fine-tuning a model like [RT-DERT](). The transformer arcitectures build that allow this zero-short prompting are usually the basis for the vision encoder in the LLM, e.g. Lama4 uses the MetaClip architecture, Gemma 3 uses SigLip and Qwen 2.5 uses ViT.
Spatial Reasoning is also not something that Gemini 2.5 introduced to LLM, it was possible with Gemini 2.0 Flash before and also open source models like Qwen2.5VL have adopted it, but given that Gemini 2.5 is crushing Humanity's Last Exam and the visual reasoning benchmark MMMU I would argue that it is the best model to check the visual reasoning capabilities of LLMs.

If you just want to quickly play around with Spatial Reasoning in LLMs you can try Google Demo [here](https://aistudio.google.com/starter-apps/spatial), but be aware that it uses Gemini 2.0 Flash not Gemini 2.5 Pro. If you want to use Gemini 2.5 Pro and also test experimental features like semantic segmentation you will need to use Googles API directly. Google provides an excellent Jupyter Notebook [here](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb) to which I will refer to for the implementation.

## How to use Geminis Spatial Reasoning most effectively

Detect big stone, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".

1. Use the experimental version of Gemini 2.0 Flash or 2.5 Pro

The first observation I made was that the default model is much worse up to the point where it often doesn't get the visual grounding right at all. The prediction are ok, but ot the location. 
The experimental version of Gemini 2.0 Flash (`gemini-2.0-flash-exp`) performs much better the default model (`gemini-2.0-flash` aka `gemini-2.0-flash-001`) and is also the version used in the WebDemo. 
For Gemini 2.5 Pro I have only access to the experimental version (`gemini-2.5-pro-exp-03-25`). 

2. Downscale the images to 640px

The higher the resolution of the image is the more the performance degrades. This is most likely due to the vision encoder used in Gemini, assuming the also used SigLip like in Gemma, then SigLip has a max resolution of 512px. 
The aistudio app downsamples the images automatically to 640px (see: https://github.com/google-gemini/starter-applets/blob/main/spatial/src/Prompt.tsx#L71) at the end I don't know what resolution will work best, but I would recommend a low one with either 640px or 512px.
While you can send the images via API in a higher resolution it will yield in worse and especially unreliable results. The results with an image size of 640px are rather reliable.
If you want to use higher resolution images I recommend cutting the image into patched and doing prediction on those patches, also known as SAHI (Slicing Aided Hyper Inference).

The combination of gemini-2.0-flash-exp and a max resolution of 640px  should work rather reliable on photographs and data similar to the COCO object detection dataset.
Yet, you might want to adopt a few points from Googles Jupyter Notebook.

3. Make the Generation Guardrails less strict. The guardrails can senitive to list and numbers and code.

```python
safety_settings = [
    genai.types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]
```

4. Use a system prompt. There are different ones depending on your task: Pointing, 2D-Boxes, 3D-Boxes & Segmentation. 
Overall Google says that 2D Boxes give the most reliable results. So I usually us them.

```python
bounding_box_system_instructions = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
If there are no relevant objects present answer with: "Nothing found"
"""
```

5. Use a temperature that is in the lower range to make the output less noisy/creative and more analytical.

```python
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=[prompt, image],
    config = genai.types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
        safety_settings=safety_settings,
    )
)
```
## Some examples - and how I got disappointed by the Spatial reasoning

Let's start with the example from Google introduction video. The shadow of the fox.

I used the following prompt:
- Detect the fox's shadow, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".
- Detect fox shadow, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".

The thing is in order to do this you do not need gemini. Other zero-shot classification models are as good in deteting this.

So I moved on to something project specific - detect the trench in this picture.
- Gemini didn't do bad
- But the grounding of some lightweight zero-shot classification model that runs on my computer is a bit better.

So I switched to another example of what I tought should be easy to detect. The stones under the orange cable. 
But what I started to realize is that with this I actually choose a hard example.
- Gemini often detected nothing sometimes something completely different, but worst of all every prediction yielded a different result, independently if pointing or 2d bounding box. This made it completely unusable.
- GroundingDino the zero-shot model I was using wasn't detecting anything, neither was Owl-ViT another zero-shot object detection model. And while that as not good either at least it was reliable.

One though was that the stones where just too small in relation to the picture, but even using patched inference didn't help. Now gemini started hallucinating and detecting stones or rocks in different spots where there where none.
The bigger problem was that it was detecting stones where there where none, which was a bigger problem that missing a few stones. 
It also turned out that Gemini 2.0 Flash was much better then the Gemini 2.5 Pro model I used.
This does not mean that 2.0 is better then 2.5, but on the Spatial Understanding it seems to be trained differently.

At this point I realized that Spatial understanding is not good enough at least for my requirements, because the only way to actually solve this issue is to fine-tune a object detection model on stone.

## Some unsatifing conclusions

Gemini Spatial Understanding is impressive. You just write a prompt and in many basic scenarios it works. No more fine-tuning of models, just some API.
However Gemini is not the only model that allows open world vocabulary zero-shot object detection Owl-Vit/Owl2 and GroundingDino are some good alternatives that are smaller, cheaper, on-device and - I would argue - as good.
Eventually the foundation models will outperform the smaller specialized models. 
But that time has not come yet. If anything I'm very disappointed in Gemini 2.5 given that Gemini 2.0 feels more stable.
And in conclusion it means at least for 2025 I still need to train computer vision models on specific datasets, which takes a lot of time.

