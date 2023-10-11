import train.trainer
class cluster:
    def __init__(self, staff_list, optimizer_list):
        self.staff_list = staff_list
        self.optimizer_list = optimizer_list
        self.staff_gl = []

    def clear(self):
        self.staff_gl.clear()

    def cluster_update(self):
        pass