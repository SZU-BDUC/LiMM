import os


class myConfig(object):
    def __init__(self, home_dir, data_dir=None, result_dir=None, trajectory_dir=None):
        self.home_dir = home_dir

        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(self.home_dir, 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if result_dir:
            self.result_dir = result_dir
        else:
            self.result_dir = os.path.join(self.home_dir, 'result')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if trajectory_dir:
            self.trajectory_dir = self.data_dir + '/' + trajectory_dir
        else:
            self.trajectory_dir = os.path.join(self.data_dir, 'trajectory')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def displayHomeDir(self):
        print("## Home direcotry: " + self.home_dir)

    def displayDataDir(self):
        print("## dataset directory: " + self.data_dir)

    def displayResultDir(self):
        print("## result directory: " + self.result_dir)

    def displayTrajectoryDir(self):
        print("## trajectory direcotry: " + self.trajectory_dir)
