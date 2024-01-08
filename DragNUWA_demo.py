import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter
import uuid
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision
from utils import *

output_dir = "outputs"
ensure_dirname(output_dir)

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points

def visualize_drag_v2(background_image_path, splited_tracks, width, height):
    trajectory_maps = []
    
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 5, (255, 0, 0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer

class Drag:
    def __init__(self, device, model_path, cfg_path, height, width, model_length):
        self.device = device
        cf = import_filename(cfg_path)
        Net, args = cf.Net, cf.args
        drag_nuwa_net = Net(args)
        state_dict = file2data(model_path, map_location='cpu')
        adaptively_load_state_dict(drag_nuwa_net, state_dict)
        drag_nuwa_net.eval()
        drag_nuwa_net.to(device)
        # drag_nuwa_net.half()
        self.drag_nuwa_net = drag_nuwa_net
        self.height = height
        self.width = width
        _, model_step, _ = split_filename(model_path)
        self.ouput_prefix = f'{model_step}_{width}X{height}'
        self.model_length = model_length

    @torch.no_grad()
    def forward_sample(self, input_drag, input_first_frame, motion_bucket_id, outputs=dict()):
        device = self.device
    
        b, l, h, w, c = input_drag.size()
        drag = self.drag_nuwa_net.apply_gaussian_filter_on_drag(input_drag)
        drag = torch.cat([torch.zeros_like(drag[:, 0]).unsqueeze(1), drag], dim=1)  # pad the first frame with zero flow
        drag = rearrange(drag, 'b l h w c -> b l c h w')

        input_conditioner = dict()
        input_conditioner['cond_frames_without_noise'] = input_first_frame
        input_conditioner['cond_frames'] = (input_first_frame + 0.02 * torch.randn_like(input_first_frame))
        input_conditioner['motion_bucket_id'] = torch.tensor([motion_bucket_id]).to(drag.device).repeat(b * (l+1))
        input_conditioner['fps_id'] = torch.tensor([self.drag_nuwa_net.args.fps]).to(drag.device).repeat(b * (l+1))
        input_conditioner['cond_aug'] = torch.tensor([0.02]).to(drag.device).repeat(b * (l+1))

        input_conditioner_uc = {}
        for key in input_conditioner.keys():
            if key not in input_conditioner_uc and isinstance(input_conditioner[key], torch.Tensor):
                input_conditioner_uc[key] = input_conditioner[key].clone()
        
        c, uc = self.drag_nuwa_net.conditioner.get_unconditional_conditioning(
            input_conditioner,
            batch_uc=input_conditioner_uc,
            force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.drag_nuwa_net.num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...")
            c[k] = repeat(c[k], "b ... -> b t ...", t=self.drag_nuwa_net.num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...")
    
        H, W = input_conditioner['cond_frames_without_noise'].shape[2:]
        shape = (self.drag_nuwa_net.num_frames, 4, H // 8, W // 8)
        randn = torch.randn(shape).to(self.device)

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            2, self.drag_nuwa_net.num_frames
        ).to(self.device)
        additional_model_inputs["num_video_frames"] = self.drag_nuwa_net.num_frames
        additional_model_inputs["flow"] = drag.repeat(2, 1, 1, 1, 1)    # c and uc

        def denoiser(input, sigma, c):
            return self.drag_nuwa_net.denoiser(self.drag_nuwa_net.model, input, sigma, c, **additional_model_inputs)
        
        samples_z = self.drag_nuwa_net.sampler(denoiser, randn, cond=c, uc=uc)
        samples = self.drag_nuwa_net.decode_first_stage(samples_z)

        outputs['logits_imgs'] = rearrange(samples, '(b l) c h w -> b l c h w', b=b)
        return outputs

    def run(self, first_frame_path, tracking_points, inference_batch_size, motion_bucket_id):
        original_width, original_height=576, 320

        input_all_points = tracking_points.constructor_args['value']
        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]

        input_drag = torch.zeros(self.model_length - 1, self.height, self.width, 2)
        for splited_track in resized_all_points:
            if len(splited_track) == 1: # stationary point
                displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[:self.model_length]
            if len(splited_track) < self.model_length:
                splited_track = splited_track + [splited_track[-1]] * (self.model_length -len(splited_track))
            for i in range(self.model_length - 1):
                start_point = splited_track[i]
                end_point = splited_track[i+1]
                input_drag[i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
                input_drag[i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]

        dir, base, ext = split_filename(first_frame_path)
        id = base.split('_')[-1]
        
        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert('RGB')
        
        visualized_drag, _ = visualize_drag_v2(first_frame_path, resized_all_points, self.width, self.height)
        
        first_frames_transform = transforms.Compose([
                        lambda x: Image.fromarray(x),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        
        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
            if not outputs:
                first_frames = image2arr(first_frame_path)
                first_frames = repeat(first_frames_transform(first_frames), 'c h w -> b c h w', b=inference_batch_size).to(self.device)
            else:
                first_frames = outputs['logits_imgs'][:, -1]
            
            outputs = self.forward_sample(
                                            repeat(input_drag[i*(self.model_length - 1):(i+1)*(self.model_length - 1)], 'l h w c -> b l h w c', b=inference_batch_size).to(self.device), 
                                            first_frames,
                                            motion_bucket_id)
            ouput_video_list.append(outputs['logits_imgs'])

        for i in range(inference_batch_size):
            ouput_tensor = [ouput_video_list[0][i]]
            for j in range(num_inference - 1):
                ouput_tensor.append(ouput_video_list[j+1][i][1:])
            ouput_tensor = torch.cat(ouput_tensor, dim=0)
            outputs_path = os.path.join(output_dir, f'output_{i}_{id}.gif')
            data2file([transforms.ToPILImage('RGB')(utils.make_grid(e.to(torch.float32).cpu(), normalize=True, range=(-1, 1))) for e in ouput_tensor], outputs_path,
                      printable=False, duration=1 / 6, override=True)

        return visualized_drag[0], outputs_path

with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">DragNUWA 1.5</h1><br>""")

    gr.Markdown("""Official Gradio Demo for <a href='https://arxiv.org/abs/2308.08089'><b>DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory</b></a>.<br>
    🔥DragNUWA enables users to manipulate backgrounds or objects within images directly, and the model seamlessly translates these actions into **camera movements** or **object motions**, generating the corresponding video.<br>
    🔥DragNUWA 1.5 enables Stable Video Diffusion to animate an image according to specific path.<br>""")

    gr.Image(label="DragNUWA", value="assets/DragNUWA1.5/Figure1.gif")

    gr.Markdown("""## Usage: <br>
                1. Upload an image via the "Upload Image" button.<br>
                2. Draw some drags.<br>
                    2.1. Click "Add Drag" when you want to add a control path.<br>
                    2.2. You can click several points which forms a path.<br>
                    2.3. Click "Delete last drag" to delete the whole lastest path.<br>
                    2.4. Click "Delete last step" to delete the lastest clicked control point.<br>
                3. Animate the image according the path with a click on "Run" button. <br>""")
    
    DragNUWA_net = Drag("cuda:0", 'models/drag_nuwa_svd.pth', 'DragNUWA_net.py', 320, 576, 14)
    first_frame_path = gr.State()
    tracking_points = gr.State([])

    def reset_states(first_frame_path, tracking_points):
        first_frame_path = gr.State()
        tracking_points = gr.State([])
        return first_frame_path, tracking_points

    def preprocess_image(image):
        image_pil = image2pil(image.name)
        raw_w, raw_h = image_pil.size
        resize_ratio = max(576/raw_w, 320/raw_h)
        image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        image_pil = transforms.CenterCrop((320, 576))(image_pil.convert('RGB'))

        first_frame_path = os.path.join(output_dir, f"first_frame_{str(uuid.uuid4())[:4]}.png")
        image_pil.save(first_frame_path)

        return first_frame_path, first_frame_path, gr.State([])

    def add_drag(tracking_points):
        tracking_points.constructor_args['value'].append([])
        return tracking_points
    
    def delete_last_drag(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map
    
    def delete_last_step(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'][-1].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map
    
    def add_tracking_points(tracking_points, first_frame_path, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        tracking_points.constructor_args['value'][-1].append(evt.index)

        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map

    with gr.Row():
        with gr.Column(scale=1):
            image_upload_button = gr.UploadButton(label="Upload Image",file_types=["image"])
            add_drag_button = gr.Button(value="Add Drag")
            reset_button = gr.Button(value="Reset")
            run_button = gr.Button(value="Run")
            delete_last_drag_button = gr.Button(value="Delete last drag")
            delete_last_step_button = gr.Button(value="Delete last step")

        with gr.Column(scale=7):
            with gr.Row():
                with gr.Column(scale=6):
                    input_image = gr.Image(label=None,
                                        interactive=True,
                                        height=320,
                                        width=576,)
                with gr.Column(scale=6):
                    output_image = gr.Image(label=None,
                                            height=320,
                                            width=576,)
    
    with gr.Row():
        with gr.Column(scale=1):
            inference_batch_size = gr.Slider(label='Inference Batch Size', 
                                             minimum=1, 
                                             maximum=1, 
                                             step=1, 
                                             value=1)
            
            motion_bucket_id = gr.Slider(label='Motion Bucket', 
                                             minimum=1, 
                                             maximum=100, 
                                             step=1, 
                                             value=4)

        with gr.Column(scale=5):
            output_video =  gr.Image(label="Output Video",
                                    height=320,
                                    width=576,)

    with gr.Row():
        gr.Markdown("""
            ## Citation
            ```bibtex
            @article{yin2023dragnuwa,
            title={Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory},
            author={Yin, Shengming and Wu, Chenfei and Liang, Jian and Shi, Jie and Li, Houqiang and Ming, Gong and Duan, Nan},
            journal={arXiv preprint arXiv:2308.08089},
            year={2023}
            }
            ```
            """)

    
    image_upload_button.upload(preprocess_image, image_upload_button, [input_image, first_frame_path, tracking_points])

    add_drag_button.click(add_drag, tracking_points, tracking_points)

    delete_last_drag_button.click(delete_last_drag, [tracking_points, first_frame_path], [tracking_points, input_image])

    delete_last_step_button.click(delete_last_step, [tracking_points, first_frame_path], [tracking_points, input_image])

    reset_button.click(reset_states, [first_frame_path, tracking_points], [first_frame_path, tracking_points])

    input_image.select(add_tracking_points, [tracking_points, first_frame_path], [tracking_points, input_image])

    run_button.click(DragNUWA_net.run, [first_frame_path, tracking_points, inference_batch_size, motion_bucket_id], [output_image, output_video])

    demo.launch(server_name="0.0.0.0", debug=True)
