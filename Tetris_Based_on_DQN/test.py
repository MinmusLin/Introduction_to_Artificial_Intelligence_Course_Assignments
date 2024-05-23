import argparse
import torch
import cv2
from src.tetris import Tetris


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--height', type=int, default=20)
    parser.add_argument('--block_size', type=int, default=30)
    parser.add_argument('--fps', type=int, default=300)
    parser.add_argument('--saved_path', type=str, default='models')
    parser.add_argument('--output', type=str, default='output.mp4')
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load('{}/model'.format(opt.saved_path))
    else:
        model = torch.load('{}/model'.format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*'MJPG'), opt.fps,
                          (int(1.5 * opt.width * opt.block_size), opt.height * opt.block_size))

    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)

        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break


if __name__ == '__main__':
    opt = get_args()
    test(opt)
