from typing import Union, List, Optional

import numpy as np
import rp
import torch
import torch.nn as nn
from easydict import EasyDict

import src.stable_diffusion as sd
from src.bilateralblur_learnabletextures import (BilateralProxyBlur,
                                    LearnableImageFourier,
                                    LearnableImageFourierBilateral,
                                    LearnableImageRaster,
                                    LearnableImageRasterBilateral)

# !!! 알파 채널도 조정가능하게 수정해야함. ----------
def make_learnable_image(height, width, num_channels, foreground=None, bilateral_kwargs=None, representation='fourier'):
    "이미지의 파라미터화 방식을 결정하여 학습 가능한 이미지를 생성."
    bilateral_kwargs = bilateral_kwargs or {}
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    if representation == 'fourier bilateral':
        return LearnableImageFourierBilateral(bilateral_blur, num_channels)
    elif representation == 'raster bilateral':
        return LearnableImageRasterBilateral(bilateral_blur, num_channels)
    elif representation == 'fourier':
        return LearnableImageFourier(height, width, num_channels)
    elif representation == 'raster':
        return LearnableImageRaster(height, width, num_channels)
    else:
        raise ValueError(f'Invalid method: {representation}')

# !!! 블렌딩 방식 변경해야함. ----------
def blend_torch_images(foreground, background, alpha):
    '주어진 foreground와 background 이미지를 alpha 값에 따라 블렌딩합니다.'
    assert foreground.shape == background.shape, 'Foreground와 background의 크기가 같아야 합니다.'
    C, H, W = foreground.shape
    assert alpha.shape == (H, W), 'alpha는 (H, W) 크기의 행렬이어야 합니다.'
    return foreground * alpha + background * (1 - alpha)

class PeekabooSegmenter(nn.Module):
    '이미지 분할을 위한 PeekabooSegmenter 클래스.'
    
    def __init__(self, 
                 image: np.ndarray, 
                 labels: List['BaseLabel'], 
                 size: int = 256,
                 channel: int = 3, 
                 name: str = 'Untitled', 
                 bilateral_kwargs: dict = None, 
                 representation: str = 'fourier bilateral', 
                 min_step=None, 
                 max_step=None):
        super().__init__()     

        self.height = self.width = size  #We use square images for now
        self.channel = channel
        self.labels = labels
        self.name = name
        self.representation = representation
        self.min_step = min_step
        self.max_step = max_step
        
        assert all(issubclass(type(label), BaseLabel) for label in labels), '모든 라벨은 BaseLabel의 서브클래스여야 합니다.'
        assert len(labels) > 0, '분할할 클래스가 최소 하나 이상 있어야 합니다.'
        
        # 이미지 전처리
        assert rp.is_image(image), '입력은 numpy 이미지여야 합니다.'
        image = rp.cv_resize_image(image, (self.height, self.width))

        # 채널에 맞게 이미지 변환
        if self.channel == 3:
            image = rp.as_rgb_image(image)  # 3채널로 변환 (RGB)
        else:
            # 입력 이미지가 3채널이 아닐 때 변환 없이 처리하도록 가정 (사용 환경에 맞게 수정 -----!!)
            assert image.shape[-1] == self.channel, f'이미지의 채널 수가 {self.channel}이 아닙니다.'

        image = rp.as_float_image(image)  # 값의 범위를 0과 1 사이로 변환
        assert image.shape == (self.height, self.width, self.channel) and image.min() >= 0 and image.max() <= 1, \
            f"이미지 크기나 값의 범위가 올바르지 않습니다. (현재 크기: {image.shape})"
        self.image = image
        
        # 이미지를 Torch 텐서로 변환 (CHW 형식)
        self.foreground = rp.as_torch_image(image).to(device)
        assert self.foreground.shape == (self.channel, self.height, self.width)
        
        # 배경은 단색으로 설정
        self.background = torch.zeros_like(self.foreground)
        
        # 학습 가능한 알파 값 생성
        bilateral_kwargs = bilateral_kwargs or {}
        self.alphas = make_learnable_image(self.height, self.width, num_channels=len(labels), 
                                           foreground=self.foreground, 
                                           representation=self.representation, 
                                           bilateral_kwargs=bilateral_kwargs)  # !!! 알파 채널도 조정가능하게 수정해야함. ----------
            
    @property
    def num_labels(self):
        '현재 레이블의 개수를 반환합니다.'
        return len(self.labels)
            
    def set_background_color(self, color):
        '배경 색상을 설정합니다. (각 채널 값은 0과 1 사이여야 함)'
        r,g,b = color
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1, "각 채널 값은 0과 1 사이여야 합니다."
        self.background[0] = r
        self.background[1] = g
        self.background[2] = b
        
    def randomize_background(self):
        '배경 색상을 무작위로 설정합니다.'
        self.set_background_color(rp.random_rgb_float_color())
        
    def forward(self, alphas=None, return_alphas=False):
        '학습된 alpha 값을 이용해 이미지 분할을 수행하고 결과 이미지를 반환합니다.'
        
        # StableDiffusion 객체의 상태를 변경
        old_min_step, old_max_step = stdf.min_step, stdf.max_step
        stdf.min_step, stdf.max_step = self.min_step, self.max_step
        
        try:
            # alpha 값이 없으면, 학습된 alpha 생성
            alphas = alphas if alphas is not None else self.alphas()
            assert alphas.shape == (self.num_labels, self.height, self.width), "alphas 크기가 맞지 않습니다."
            assert alphas.min() >= 0 and alphas.max() <= 1, "alphas 값은 0과 1 사이여야 합니다."

            # alpha 값을 이용하여 각 라벨에 대한 이미지를 생성
            output_images = [blend_torch_images(self.foreground, self.background, alpha) for alpha in alphas]
            output_images = torch.stack(output_images)
            assert output_images.shape == (self.num_labels, self.channel, self.height, self.width), "출력 이미지 크기가 맞지 않습니다."

            return (output_images, alphas) if return_alphas else output_images

        finally:
            # StableDiffusion 객체의 원래 상태 복원
            stdf.min_step, stdf.max_step = old_min_step, old_max_step

def display(self):  # !!! 이거 배경색 입히지 말고 그냥 없애는 걸로 변경해야함 -----------.
    'PeekabooSegmenter의 이미지를 다양한 배경색과 함께 시각화하는 메서드.'

    # 기본 색상 설정 및 랜덤 색상 생성
    assert self.channel == 1 or 3, '채널이 1 또는 3이 아님.'
    if self.channel == 3:
        colors = [rp.random_rgb_float_color() for _ in range(3)]
    elif self.channel == 1:
        colors = [[0, 0, 0]]
    alphas = rp.as_numpy_array(self.alphas())
    assert alphas.shape == (self.num_labels, self.height, self.width)

    # 배경색과 함께 각 알파 채널로 생성된 이미지를 저장 -> 이미지는 '[[i1], [i2], [i3]]' 이런 형태
    composites = [rp.as_numpy_images(self(self.alphas())) for color in colors for _ in [self.set_background_color(color)]]

    # 레이블 이름 및 상태 정보 설정
    label_names = [label.name for label in self.labels]
    stats_lines = [self.name, '', f'H,W = {self.height}x{self.width}']

    # 전역 변수에서 특정 상태 정보를 추가
    for stat_format, var_name in [('Gravity: %.2e', 'GRAVITY'),
                                    ('Batch Size: %i', 'BATCH_SIZE'),
                                    ('Iter: %i', 'iter_num'),
                                    ('Image Name: %s', 'image_filename'),
                                    ('Learning Rate: %.2e', 'LEARNING_RATE'),
                                    ('Guidance: %i%%', 'GUIDANCE_SCALE')]:
        if var_name in globals():
            stats_lines.append(stat_format % globals()[var_name])

    # 이미지와 알파 채널을 각 배경색과 함께 결합하여 출력 이미지 생성
    output_image = rp.labeled_image(
        rp.tiled_images(
            rp.labeled_images(
                [self.image,
                    alphas[0],
                    composites[0][0],
                    composites[1][0] if len(composites) > 1 else None,
                    composites[2][0] if len(composites) > 2 else None],
                ["Input Image",
                    "Alpha Map",
                    "Background #1",
                    "Background #2" if len(composites) > 1 else None,
                    "Background #3" if len(composites) > 2 else None],),
            length=2 + len(composites),),
        label_names[0])

    # 이미지 출력
    rp.display_image(output_image)

    return output_image

PeekabooSegmenter.display=display

def get_mean_embedding(prompts:list):
    '주어진 프롬프트 리스트의 평균 임베딩을 계산하여 반환합니다'
    return torch.mean(
        torch.stack([stdf.get_text_embeddings(prompt) for prompt in prompts]),
        dim=0
    ).to(device)

class BaseLabel:
    '기본 레이블 클래스. 이름과 임베딩을 저장하며, 샘플 이미지를 생성할 수 있습니다'
    def __init__(self, name:str, embedding:torch.Tensor):
        #Later on we might have more sophisticated embeddings, such as averaging multiple prompts
        #We also might have associated colors for visualization, or relations between labels
        self.name=name
        self.embedding=embedding
        
    def get_sample_image(self):
        '임베딩을 기반으로 샘플 이미지를 생성하여 반환합니다.'
        output = stdf.embeddings_to_imgs(self.embedding)[0]
        assert rp.is_image(output), '생성된 출력이 이미지가 아닙니다.'
        return output

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"
        
class SimpleLabel(BaseLabel):
    '텍스트 임베딩을 사용한 간단한 레이블 클래스.'
    def __init__(self, name:str):
        super().__init__(name, stdf.get_text_embeddings(name).to(device))

class MeanLabel(BaseLabel):
    '여러 프롬프트의 평균 임베딩을 사용한 레이블 클래스'
    #Test: rp.display_image(rp.horizontally_concatenated_images(MeanLabel('Dogcat','dog','cat').get_sample_image() for _ in range(1)))
    def __init__(self, name:str, *prompts):
        super().__init__(name, get_mean_embedding(rp.detuple(prompts)))

class PeekabooResults(EasyDict):
    'dict처럼 동작하지만 속성처럼 읽고 쓸 수 있는 클래스.'
    pass

def save_peekaboo_results(results,new_folder_path):
    'PeekabooResults를 지정된 폴더에 저장합니다.'

    import json
    assert not rp.folder_exists(new_folder_path), f'Please use a different name, not {new_folder_path}'
    rp.make_folder(new_folder_path)

    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print(f"\nSaving PeekabooResults to {new_folder_path}")
        params = {}

        for key, value in results.items():
            if rp.is_image(value):
                rp.save_image(value, f'{key}.png')  # 단일 이미지 저장
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):
                rp.make_directory(key)  # 이미지 폴더 저장
                with rp.SetCurrentDirectoryTemporarily(key):
                    [rp.save_image(img, f'{i}.png') for i, img in enumerate(value)]
            elif isinstance(value, np.ndarray):
                np.save(f'{key}.npy', value)  # 일반 Numpy 배열 저장
            else:
                try:
                    json.dumps({key: value})  # JSON으로 변환 가능한 값 저장
                    params[key] = value
                except Exception:
                    params[key] = str(value)  # 변환 불가한 값은 문자열로 저장

        rp.save_json(params, 'params.json', pretty=True)
        print(f"Done saving PeekabooResults to {new_folder_path}!")
        
def make_image_square(image: np.ndarray, method='crop') -> np.ndarray:  # !!! 이것도 3채널인거 가정하고 하는듯. 수정 필요 -------
    """
    주어진 이미지를 512x512(x3) 크기의 정사각형으로 변환합니다. 
    method는 'crop' 또는 'scale' 중 하나를 사용할 수 있습니다.
    """
    assert rp.is_image(image)
    assert method in ['crop','scale']
    
    try:
        image = rp.as_rgb_image(image)  # 이미지가 3채널 RGB인지 확인 및 변환
        height, width = rp.get_image_dimensions(image)
        min_dim = min(height, width)
    except:
        height, width = rp.get_image_dimensions(image)
        min_dim = min(height, width)

    if method == 'crop':
        # 중앙에서 자르고 스케일링하여 정사각형 이미지 생성
        return make_image_square(rp.crop_image(image, min_dim, min_dim, origin='center'), 'scale')
    
    # 'scale' 메서드일 경우 이미지 크기를 512x512로 조정
    return rp.resize_image(image, (512, 512))

def run_peekaboo(name: str,
                 image: Union[str, np.ndarray],
                 label: Optional['BaseLabel'] = None,

                 GRAVITY=1e-1/2,
                 NUM_ITER=300,
                 LEARNING_RATE=1e-5, 
                 BATCH_SIZE=1,   
                 GUIDANCE_SCALE=100,
                 bilateral_kwargs=dict(
                     kernel_size=3,
                     tolerance=0.08,
                     sigma=5,
                     iterations=40
                     ),
                 square_image_method='crop', 
                 representation='fourier bilateral',
                 min_step=None, 
                 max_step=None) -> PeekabooResults:
    """
    Peekaboo Hyperparameters:
    GRAVITY=1e-1/2: prompt에 따라 tuning이 제일 필요함. 주로 1e-2, 1e-1/2, 1e-1, or 1.5*1e-1에서 잘 됨.
    NUM_ITER=300: 300이면 대부분 충분
    LEARNING_RATE=1e-5: neural neural textures 아닐 경우, 값 키워도 됨
    BATCH_SIZE=1: 큰 차이 없음. 배치 키우면 vram만 잡아먹음
    GUIDANCE_SCALE=100: DreamFusion 논문의 고정 값임.
    bilateral_kwargs=(kernel_size=3,tolerance=.08,sigma=5,iterations=40)
    square_image_method: input image를 정사각형화 하는 두 가지 방법. (crop / scale)
    representation: (fourier bilateral / raster bilateral / fourier / raster)
    """
    
    # 레이블이 없을 경우 기본 레이블 생성
    label = label or SimpleLabel(name)

    # 이미지 로드 및 전처리
    image_path = image if isinstance(image, str) else '<No image path given>'
    image = rp.load_image(image_path) if isinstance(image, str) else image  # 내가 임의로 image_path로 바꾼거긴해..00
    assert rp.is_image(image)
    assert issubclass(type(label), BaseLabel)
    image = rp.as_rgb_image(rp.as_float_image(make_image_square(image, square_image_method)))

    rp.tic()
    time_started=rp.get_current_date()
    
    # PeekabooSegmenter 생성
    p=PeekabooSegmenter(image, labels=[label], name=name, 
                        bilateral_kwargs=bilateral_kwargs, 
                        representation=representation, 
                        min_step=min_step, 
                        max_step=max_step).to(device)

    if 'bilateral' in representation:
        blur_image = rp.as_numpy_image(p.alphas.bilateral_blur(p.foreground))
        print("The bilateral blur applied to the input image before/after, to visualize it")
        rp.display_image(rp.tiled_images(rp.labeled_images([rp.as_numpy_image(p.foreground), blur_image], ['before', 'after'])))

    p.display()

    # 옵티마이저 설정
    params = list(p.parameters())
    optim = torch.optim.SGD(params, lr=LEARNING_RATE)

    # 학습 반복 설정
    global iter_num
    iter_num = 0
    timelapse_frames=[]
    preview_interval = max(1, NUM_ITER // 10)  # 10번의 미리보기를 표시

    try:
        display_eta = rp.eta(NUM_ITER)
        for _ in range(NUM_ITER):
            display_eta(_)
            iter_num += 1

            alphas = p.alphas()
            for __ in range(BATCH_SIZE):
                p.randomize_background()
                composites = p()
                for label, composite in zip(p.labels, composites):
                    stdf.train_step(label.embedding, composite[None], guidance_scale=GUIDANCE_SCALE)

            ((alphas.sum()) * GRAVITY).backward()
            optim.step()
            optim.zero_grad()

            with torch.no_grad():
                if not _ % preview_interval: 
                    timelapse_frames.append(p.display())

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")
                

    results = PeekabooResults(
        #The main output
        alphas=rp.as_numpy_array(alphas),
        
        #Keep track of hyperparameters used
        GRAVITY=GRAVITY, BATCH_SIZE=BATCH_SIZE, NUM_ITER=NUM_ITER, GUIDANCE_SCALE=GUIDANCE_SCALE,
        bilateral_kwargs=bilateral_kwargs, representation=representation, label=label,
        image=image, image_path=image_path, 
        
        #Record some extra info
        preview_image=p.display(), timelapse_frames=rp.as_numpy_array(timelapse_frames),
        **({'blur_image':blur_image} if 'blur_image' in dir() else {}),
        height=p.height, width=p.width, p_name=p.name, min_step=p.min_step, max_step=p.max_step,
        
        # git_hash=rp.get_current_git_hash(), 
        time_started=rp.r._format_datetime(time_started),
        time_completed=rp.r._format_datetime(rp.get_current_date()),
        device=device, computer_name=rp.get_computer_name()) 
    
    # 결과 폴더 생성 및 저장
    output_folder = rp.make_folder(f'peekaboo_results/{name}')
    output_folder += f'/{len(rp.get_subfolders(output_folder)):03}'
    save_peekaboo_results(results, output_folder)

    # # 학습 진행 타임랩스 표시
    # print("Please wait - creating a training timelapse")
    # # clear_output() ----------- Ipython 쓸때만!
    # rp.display_image_slideshow(timelapse_frames)
    # print(f"Saved results at {output_folder}")

    return results
  
TOKEN='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'
#Importing this module loads a stable diffusion model. Hope you have a GPU!
stdf=sd.StableDiffusion(TOKEN, 'cuda','CompVis/stable-diffusion-v1-4', variant="fp16")
device=stdf.device