#import library 
import argparse
from detect_action import detect_action_webcam
import tensorflow as tf

# for running realtime detection
def run_realtime():
    detect_action_webcam()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("func_name", type=str,
                        help="Select a function to run. <emo_realtime> or <emo_path>")
    # parse the args
    args = parser.parse_args()

    #print('****ARGS: ' + str(args))

    if args.func_name == "realtime":
        run_realtime()
    else:
        print("Usage: python main.py <function name>")

if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    # Place tensors on the CPU
    with tf.device('/CPU:0'):
        main()