---
title: 'Gemini 2.5 Spatial Reasoning'
date: 2025-04-09T09:23:00+02:00
tags: ['Gemini-2.5', 'Gemini']
summary: "Testing the limits of Gemini 2.5 Spatial Reasoning capabilities on the example of fiber construction trenches"
---

# Gemini Spatial Reasoning Is Amazing  
**...but not good enough to be helpful yet.**

Various large language models with vision support enable users to ask questions about images, usually referred to as [visual reasoning]. This allows queries like: *How many pumpkins do you see in this picture?* or *What kind of flower is in this picture?* This is already very useful, as it enables zero-shot classification without needing to train a model.  

However, when you need to determine **where** that pumpkin or flower is in the picture, it's not possible—yet. That’s where **spatial reasoning** comes in, allowing models to return bounding boxes or coordinates for objects through prompting.

Gemini 2.5 is not the first model to enable this. For years, most vision models have been built on top of transformer architectures and allow zero-shot prompting to some degree, like MobileCLIP, SigLIP v2, or GroundingDINO—the latter even enabling bounding boxes, not just classification.  

Additionally, visual grounding was possible with other vision-supporting LLMs like GPT-4o, but building such pipelines was complex and, in my opinion, not worth the effort compared to fine-tuning a model like [RT-DETR](https://arxiv.org/abs/2407.17140).

The transformer architectures that make this possible are typically the basis of the vision encoder in LLMs. For example:  
- LLaMA 4 uses the MetaCLIP architecture  
- Gemma 3 uses SigLIP  
- Qwen 2.5 uses ViT

Spatial reasoning is not something Gemini 2.5 introduced to LLMs—it was already available in Gemini 2.0 Flash and in open-source models like Qwen2.5-VL. However, given that Gemini 2.5 is dominating benchmarks like *HumanEval* and *MMMU*, I’d argue it's the best model for testing visual reasoning in LLMs today.

If you just want to play around with spatial reasoning, try the Google demo [here](https://aistudio.google.com/starter-apps/spatial)—but note that it uses Gemini 2.0 Flash, not 2.5 Pro. If you want to use Gemini 2.5 Pro and test experimental features like semantic segmentation, you’ll need to use Google’s API directly. Google provides an excellent Jupyter Notebook [here](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb), which I’ll reference for implementation tips.

---

## How to Use Gemini’s Spatial Reasoning Most Effectively

The visual grounding of Gemini is still unstable and will yield different results when prompted differently as well as with every request there is some randomness to it. In order for it to work somewhat reliable you need to set the following things.

1. A proper user prompt and ans additional system prompt that contain some of the trigger words that result in spatial json output
2. Some time to prompt engineer you actual request to get reliable results


{{< figure src="socks_gemini.png" alt="Prediction of Gemini 2.0 Flash" caption="Gemini 2.0 Flash" >}}

> Detect rainbow sock, with no more than 20 items. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".

For the image above that user prompt from Google worked and I usually use it as a template. But to actually get reliable results there are a few things I would recommend:

### 1. Use the experimental version of Gemini 2.0 Flash or 2.5 Pro

One of my first observations: the default model 2.0 Flash model performs much worse. While the predictions of something being present might be okay, it often fails at the visual grounding. In other words the *bounding boxes* are wrong in the wrong place.

The experimental Gemini 2.0 Flash (`gemini-2.0-flash-exp`) performs significantly better than the default model (`gemini-2.0-flash`, aka `gemini-2.0-flash-001`) and is also the version used in the web demo.  

For Gemini 2.5 Pro, I only had access to the experimental version: `gemini-2.5-pro-exp-03-25`.

### 2. Downscale images to 640px

Higher image resolutions lead to worse performance—likely due to the vision encoder limits. Assuming Gemini also uses SigLIP (as used in Gemma 3), that model has a max resolution of 512px.

The Aistudio app downscales images to 640px automatically ([see here](https://github.com/google-gemini/starter-applets/blob/main/spatial/src/Prompt.tsx#L71)).  
I’m not sure what the optimal resolution is, but I’d recommend sticking to 512px or 640px.

Sending higher-res images via the API results in **worse grounding** and **more unreproducable** outputs. 640px works well and yields consistent results.  
If you need higher detail, consider splitting the image into patches and doing prediction on each—this technique is known as **SAHI (Slicing Aided Hyper Inference).**

{{< figure src="upscale.png" alt="Slicing Aided Hyper Inference" caption="Simplified Slicing Aided Hyper Inference" >}}


The combo of `gemini-2.0-flash-exp` and 640px resolution performs the reliable in my test - at least on photos and data similar to the COCO dataset. Still, I recommend adapting some ideas from Google's notebook.

### 3. Loosen the generation guardrails

The default safety filters are often too sensitive to this kind of output. So you can make them less stict like this:

```python
safety_settings = [
    genai.types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]
```

### 4. Use a task-specific system prompt

There are different system prompts for:  
- Pointing  
- 2D boxes  
- 3D boxes  
- Segmentation  

Google suggests 2D boxes give the most reliable results. I usually go with that. The following is the system prompt I used for my tests.

```python
bounding_box_system_instructions = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
If there are no relevant objects present answer with: "Nothing found"
"""
```

If one visual ground does not work you can test a different approach. The system prompts for the other task you can also find in Googles Notebook

### 5. Use a low temperature (<= 0.5)

Lower temperatures make the output less creative and more consistent. If you combine all of this you should end up with a API request like this:

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

---

## Some Examples—and Why I Got Disappointed

Let’s start with the example from [Google’s introduction video](https://www.youtube.com/watch?v=-XmoDzDMqj4): **the fox’s shadow**.

> Detect the fox's shadow, with no more than 20 items. Output a JSON list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".

It matters a bit how you prompt it because if you ask "Detect fox shadow,..." it will not work. It only works if you ask "Detect the fox's shadow, ..." in that way the results can be a bit unpredictable.

{{< figurerow >}}
  {{< figure src="fox_gemini.png" alt="Prediction of Gemini 2.0 for the Prompt: the fox's shadow" caption="Gemini 2.0 - Prompt: the fox's shadow" >}}
  {{< figure src="fox_image.jpg" alt="Prediction of Gemini 2.0 for the Prompt: the fox shadow" caption="Gemini 2.0 - Prompt: fox shadow" >}}
  {{< figure src="fox_groundingdino.jpg" alt="Prediction of GroundingDino for the Prompt: fox shadow" caption="GroundingDino - Prompt: fox shadow" >}}
{{</figurerow >}}

Here’s the thing—you don’t need Gemini for this. Other zero-shot models perform just as well at identifying it. Above prediction was done by [GroundingDino](https://huggingface.co/IDEA-Research/grounding-dino-base) a 172M parameter model.

So I tried something more project-specific: **Detecting a trench**

{{< figurerow >}}
  {{< figure src="trench_gemini.png" alt="Prediction of Gemini 2.0 for the Prompt: trench" caption="Gemini 2.0 - Prompt: trench" >}}
  {{< figure src="trench_groundingdino.jpg" alt="Prediction of GroundingDino for the Prompt: trench" caption="GroundingDino - Prompt: a long narrow ditch" >}}
{{</figurerow >}}
{{< figurerow >}}
  {{< figure src="trench2_gemini.png" alt="Prediction of Gemini 2.0 for the Prompt: trench" caption="Gemini 2.0 - Prompt: trench" >}}
  {{< figure src="trench2_groundingdino.jpg" alt="Prediction of GroundingDino for the Prompt: trench" caption="GroundingDino - Prompt: a long narrow ditch" >}}
{{</figurerow >}}

Gemini didn’t do badly it's easier to prompt then lightweight zero-shot model GroundingDino. GroundingDino was not working well with just the term trench, but overall it then had slightly better visual grounding.

Next, I tried something that *should* have been easy: **Detecting the stones under an orange cable**. 
Turns out—it wasn’t.

As reference if I just ask Gemini about it, it will give me the following anwer which is correct:

>Yes, there is gravel and some larger stones visible in the image, particularly within the excavated trench area around the orange pipes. The soil dug out appears to be a mix of earth and rock fragments of various sizes, consistent with gravel and possibly some stones larger than typical gravel.

{{< figurerow >}}
  {{< figure src="stones_annotation.png" alt="Annotation" caption="Annotation" >}}
  {{< figure src="stones_gemini20_1.png" alt="Gemini 2.0 Flash" caption="Gemini 2.0 Flash" >}}
  {{< figure src="stones_gemini25_0.png" alt="Gemini 2.5 Pro" caption="Gemini 2.5 Pro for stones" >}}
  {{< figure src="stones_gemini25_1.png" alt="Gemini 2.5 Pro" caption="Gemini 2.5 Pro for gravel" >}}
  {{< figure src="stones_gemini25_2.png" alt="Gemini 2.5 Pro" caption="Gemini 2.5 Pro for biggest stone" >}}
  {{< figure src="stones_gemini_20_pointing.png" alt="Gemini 2.0 Flash" caption="Gemini 2.0 Flash pointing" >}}
{{</figurerow >}}

Gemini either detected nothing or hallucinated irrelevant objects. The main problem where that the results where different on every run. Switching form 2D boxes to pointing didn't help. GroundingDINO and OWLv2 also failed, but at least they were consistently failing, which is arguably better than hallucinating. Areas that had no stones often had wrong predictions specifically with 2.0 Flash, where the mistakes where more obvious.

{{< figurerow >}}
  {{< figure src="notstones_gemini20.png" alt="Gemini 2.0 Flash" caption="Gemini 2.0 Flash SAHI" >}}
  {{< figure src="stones_zoom_gemini25.png" alt="Gemini 2.5 Pro" caption="Gemini 2.5 Pro SAHI" >}}
{{</figurerow >}}


I thought maybe the stones were too small, but patch-based (SAHI) inference didn’t help either.  
In fact, Gemini started hallucinating. Detecting even more stones where there were none, which is a bigger issue than missing a few.


At this point, I realized that for my needs, Geminis spatial reasoning isn’t good enough. The only real solution would be to fine-tune an object detection model for the specific class “stone”.

---

## Some Unsatisfying Conclusions

Gemini’s spatial reasoning is impressive. You write a prompt—and it just works, for simple use-cases. No model training, just API calls.  

But Gemini isn’t the only model supporting zero-shot, open-world object detection.  
**OWL-ViT, OWL2, and GroundingDINO** are good alternatives: smaller, cheaper, and runnable on-device.  
And I’d argue they’re just as good as Gemini for zero-shot object detection or other spatial reasoning tasks. Only on complex reasoning Gemini has the edge, eg. if you query is very complex. I left being a bit disappointed by Gemini 2.5—especially given that the improvement over the spatial reasoning from Gemini 2.0 Flash is that minor.

Eventually, foundation models will surpass smaller, fine-tuned models. But that time isn’t here yet.  

So for now, in 2025, I still need to train CV models on specific datasets, which is tedious—but necessary.
