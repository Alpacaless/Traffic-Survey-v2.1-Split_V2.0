import threading
import UVI.SecondLevelFunc.analysisFunction as analysisFunction
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)

def start_run_thread(opt):
    global current_run_thread, stop_run_flag
    stop_run_flag = result['stop_run_flag']
    current_run_thread = threading.Thread(target=analysisFunction.analysisFunction,name='MainCode', args=(opt,))
    current_run_thread.start()
    print("run() 线程启动完成")
