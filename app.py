import gradio as gr
from urllib.parse import urlparse
import requests
import time
import os
import re
from gradio_client import Client

is_shared_ui = True if "fffiloni/consistent-character" in os.environ['SPACE_ID'] else False
def safety_check(user_prompt, token):
    
    client = Client("fffiloni/safety-checker-bot", hf_token=token)
    response = client.predict(
            source_space="consistent-character space",
            user_prompt=user_prompt,
            api_name="/infer"
    )
    print(response)
    
    return response
    
from utils.gradio_helpers import parse_outputs, process_outputs

names = ['prompt', 'negative_prompt', 'subject', 'number_of_outputs', 'number_of_images_per_pose', 'randomise_poses', 'output_format', 'output_quality', 'seed']

def predict(request: gr.Request, *args, progress=gr.Progress(track_tqdm=True)):
    print(f"""
        —/n
        {args[0]}
        """)
    if args[0] == '' or args[0] is None:
        raise gr.Error(f"You forgot to provide a prompt.")
    
    try:
        if is_shared_ui:
            hf_token = os.environ.get("HF_TOKEN")

            is_safe = safety_check(args[0], hf_token)
            print(is_safe)
    
            match = re.search(r'\bYes\b', is_safe)
    
            if match:
                status = 'Yes'
            else:
                status = None
        else:
            status = None
    
        if status == "Yes" :
            raise gr.Error("Do not ask for such things.")
        else:
    
            headers = {'Content-Type': 'application/json'}
    
            payload = {"input": {}}
        
        
            base_url = "http://0.0.0.0:7860"
            for i, key in enumerate(names):
                value = args[i]
                if value and (os.path.exists(str(value))):
                    value = f"{base_url}/gradio_api/file=" + value
                if value is not None and value != "":
                    payload["input"][key] = value
    
            response = requests.post("http://0.0.0.0:5000/predictions", headers=headers, json=payload)
    
        
            if response.status_code == 201:
                follow_up_url = response.json()["urls"]["get"]
                response = requests.get(follow_up_url, headers=headers)
                while response.json()["status"] != "succeeded":
                    if response.json()["status"] == "failed":
                        raise gr.Error("The submission failed!")
                    response = requests.get(follow_up_url, headers=headers)
                    time.sleep(1)
            if response.status_code == 200:
                json_response = response.json()
                #If the output component is JSON return the entire output response 
                if(outputs[0].get_config()["name"] == "json"):
                    return json_response["output"]
                predict_outputs = parse_outputs(json_response["output"])
                processed_outputs = process_outputs(predict_outputs)        
                return tuple(processed_outputs) if len(processed_outputs) > 1 else processed_outputs[0]
            else:
                if(response.status_code == 409):
                    raise gr.Error(f"Sorry, the Cog image is still processing. Try again in a bit.")
                raise gr.Error(f"The submission failed! Error: {response.status_code}")

    except Exception as e:
        # Handle any other type of error
        raise gr.Error(f"An error occurred: {e}")

title = "Demo for consistent-character cog image by fofr"
description = "Create images of a given character in different poses • running cog image by fofr"

css="""
#col-container{
    margin: 0 auto;
    max-width: 1400px;
    text-align: left;
}
"""
with gr.Blocks(css=css) as app:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Consistent Character Workflow")
        gr.Markdown("### Create images of a given character in different poses • running cog image by fofr")
        
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://huggingface.co/spaces/fffiloni/consistent-character?duplicate=true">
				<img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
			</a>
            <p> to skip the queue and use custom prompts
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                if is_shared_ui:
                    prompt = gr.Textbox(
                        label="Prompt", info='''Duplicate the space to you personal account for custom prompt''',
                        value="a person, darkblue suit, black tie, white pocket",
                        interactive=False
                    )
                else:
                    prompt = gr.Textbox(
                        label="Prompt", info='''Describe the subject. Include clothes and hairstyle for more consistency.''',
                        value="a person, darkblue suit, black tie, white pocket",
                        interactive=True
                    )
        
                subject = gr.Image(
                    label="Subject", type="filepath"
                )

                submit_btn = gr.Button("Submit")

                with gr.Accordion(label="Advanced Settings", open=False):
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt", info='''Things you do not want to see in your image''',
                        value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
                    )

                    with gr.Row():

                        number_of_outputs = gr.Slider(
                            label="Number Of Outputs", info='''The number of images to generate.''', value=4,
                            minimum=1, maximum=4, step=1,
                        )
                        
                        number_of_images_per_pose = gr.Slider(
                            label="Number Of Images Per Pose", info='''The number of images to generate for each pose.''', value=1,
                            minimum=1, maximum=4, step=1,
                        )

                    with gr.Row():
                        
                        randomise_poses = gr.Checkbox(
                            label="Randomise Poses", info='''Randomise the poses used.''', value=True
                        )
                        
                        output_format = gr.Dropdown(
                            choices=['webp', 'jpg', 'png'], label="output_format", info='''Format of the output images''', value="webp"
                        )
                    
                    with gr.Row():
                        
                        output_quality = gr.Number(
                            label="Output Quality", info='''Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.''', value=80
                        )
                        
                        seed = gr.Number(
                            label="Seed", info='''Set a seed for reproducibility. Random by default.''', value=None
                        )

            with gr.Column(scale=3):
                consistent_results = gr.Gallery(label="Consistent Results")

    inputs = [prompt, negative_prompt, subject, number_of_outputs, number_of_images_per_pose, randomise_poses, output_format, output_quality, seed]
    outputs = [consistent_results]

    submit_btn.click(
        fn = predict,
        inputs = inputs,
        outputs = outputs,
        show_api = False
    )

app.queue(max_size=12, api_open=False).launch(share=False, show_api=False, show_error=True)

