# src/generate_cli.py
from prompts import split_story_to_sentences, build_prompt
from generate import generate_images, save_comic_strip
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", type=str, default="A boy finds a glowing key in the forest. He opens a chest and finds treasure.", help="Short story")
    parser.add_argument("--style", type=str, default="anime", help="Style: anime/realistic/cartoon")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--cols", type=int, default=2, help="Panels per row")
    args = parser.parse_args()

    panels = split_story_to_sentences(args.story, max_panels=4)
    prompts = [build_prompt(p, style=args.style) for p in panels if p.strip()]
    print("Prompts:")
    for i, p in enumerate(prompts):
        print(f"Panel {i+1}: {p}")
    images = generate_images(prompts, num_inference_steps=args.steps, guidance_scale=args.guidance, seed=args.seed)
    out = save_comic_strip(images, out_path="examples/demo_comic.png", cols=args.cols)
    print(f"Saved comic to {out}")

if __name__ == "__main__":
    main()
