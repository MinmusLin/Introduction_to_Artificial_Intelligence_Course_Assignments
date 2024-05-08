import argparse
import torch
import warnings
from torch.serialization import SourceChangeWarning
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

warnings.filterwarnings("ignore", category=SourceChangeWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--saved_path', type=str, default='models')
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2250758)
    else:
        torch.manual_seed(2250758)

    if torch.cuda.is_available():
        model = torch.load('{}/model'.format(opt.saved_path))
    else:
        model = torch.load('{}/model'.format(opt.saved_path), map_location=lambda storage, loc: storage)

    model.eval()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()

    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        state = next_state


if __name__ == '__main__':
    option = get_args()
    test(option)
