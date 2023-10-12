import train.trainer
class cluster:
    def __init__(self, staff_list, optimizer_list):
        self.staff_list = staff_list
        self.optimizer_list = optimizer_list
        self.staff_gl = []
        self.mean_gl = []
    def clear(self):
        self.staff_gl.clear()
        self.mean_gl.clear()
    def compute_grad_gl(self, metric):
        for staff in self.staff_list:
            gl_dict = staff.compute_grad(metric=metric)
            self.staff_gl.append(gl_dict)
        return self.staff_gl
    def compute_grad_mean(self):
        pass