import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "RB-2003/multilingual-factual-consistency-ties"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)

model.to(device)
model.eval()

@torch.no_grad()
def generate(prompt, max_new_tokens):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(1, 50, value=15, step=1, label="Max new tokens"),
    ],
    outputs=gr.Textbox(label="Edited model output"),
    title="Multilingual Factual Consistency — MEMIT Demo",
)

if __name__ == "__main__":
    demo.launch()
