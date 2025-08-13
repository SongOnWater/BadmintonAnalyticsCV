import argparse
import os
import sys
import subprocess
import time


def run_pre_predict(video_file: str, save_dir: str) -> float:
    script_path = os.path.join(os.path.dirname(__file__), 'pre_predict.py')
    cmd = [sys.executable, script_path, '--video_file', video_file, '--save_dir', save_dir]
    t0 = time.time()
    completed = subprocess.run(cmd, check=True)
    t1 = time.time()
    return t1 - t0


def run_testing() -> float:
    script_path = os.path.join(os.path.dirname(__file__), 'testing.py')
    cmd = [sys.executable, script_path]
    t0 = time.time()
    completed = subprocess.run(cmd, check=True)
    t1 = time.time()
    return t1 - t0


def main() -> None:
    parser = argparse.ArgumentParser(description='Run full badminton analytics pipeline (predict + scoring video).')
    parser.add_argument('--video_file', required=True, type=str, help='Input video path')
    parser.add_argument('--save_dir', default='prediction', type=str, help='Directory to save outputs')
    args = parser.parse_args()

    video_file = os.path.abspath(args.video_file)
    save_dir = os.path.abspath(args.save_dir)

    if not os.path.exists(video_file):
        print(f"Input video not found: {video_file}")
        sys.exit(1)

    os.makedirs(save_dir, exist_ok=True)

    overall_start = time.time()
    pre_predict_duration = 0.0
    testing_duration = 0.0
    try:
        print('Step 1/2: Running pre_predict...')
        pre_predict_duration = run_pre_predict(video_file=video_file, save_dir=save_dir)
        print(f"pre_predict finished in {pre_predict_duration:.2f}s")

        print('Step 2/2: Running testing (score overlay and clip generation)...')
        testing_duration = run_testing()
        print(f"testing finished in {testing_duration:.2f}s")
    except subprocess.CalledProcessError as e:
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        print(f"Pipeline failed with return code {e.returncode}")
        print(f"Elapsed before failure: total {overall_duration:.2f}s | pre_predict {pre_predict_duration:.2f}s | testing {testing_duration:.2f}s")
        sys.exit(e.returncode)

    # Summarize outputs
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')
    out_score_clip = os.path.join(save_dir, f'{video_name}_score_clip.mp4')

    overall_end = time.time()
    overall_duration = overall_end - overall_start

    print('\nOutputs:')
    print(f"- CSV: {out_csv_file} {'(found)' if os.path.exists(out_csv_file) else '(missing)'}")
    print(f"- Predicted video: {out_video_file} {'(found)' if os.path.exists(out_video_file) else '(missing)'}")
    print(f"- Score clip: {out_score_clip} {'(found)' if os.path.exists(out_score_clip) else '(missing)'}")
    print(f"\nDurations: total {overall_duration:.2f}s | pre_predict {pre_predict_duration:.2f}s | testing {testing_duration:.2f}s")


if __name__ == '__main__':
    main()


