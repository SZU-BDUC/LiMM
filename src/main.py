import os
import argparse
from utils.config import myConfig
import my_experiment as epm
from ReinforceLearningRadius import UseReinforceLearnedRadius


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':

    home_dir = os.path.join(os.path.dirname(os.getcwd()))

    # python main.py porto -p trajectory -e HMMLimm
    parser = argparse.ArgumentParser(description="experiment for mapmatching", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("name", help="dataset name")
    parser.add_argument("-s", "--data_dir", help="dataset directory")
    parser.add_argument("-d", "--result_dir", help="destination result directory")
    parser.add_argument("-p", "--trajectory_dir", help="trajectory data directory")
    parser.add_argument("-e", "--experiment_name", help="HMMLimm")
    args = parser.parse_args()
    parm = vars(args)
    name = parm['name']
    if parm['data_dir']:
        data_dir = parm['data_dir']
    else:
        data_dir = home_dir + "/data/" + name
    if parm['result_dir']:
        result_dir = parm['result_dir']
    else:
        result_dir = home_dir + "/result/" + name
    if parm['trajectory_dir']:
        trajectory_dir = parm['trajectory_dir']
    else:
        trajectory_dir = 'trajectory'

    ## print running log
    print("## Running experiment on dataset '" + name + "'")
    print('-' * 50)
    print(trajectory_dir.split('/')[-1])

    config = myConfig(home_dir, data_dir, result_dir, trajectory_dir)
    config.displayHomeDir()
    config.displayDataDir()
    config.displayResultDir()
    config.displayTrajectoryDir()

    ## get road network
    print('-' * 50)
    print('road network processing')
    get_seg = epm.GetSegments(config, name)
    print('road network processing finish')
    print('-' * 50)


    if parm['experiment_name'] == 'HMMLimm':
        print('using Limm to match')
        experiment = epm.HMMLimm(config)
        experiment.run(50)
    else:
        print('need to check experiment name')


