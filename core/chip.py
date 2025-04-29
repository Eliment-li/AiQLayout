#representing a quantum computing device chip
class Chip():
    def __init__(self, name):
        self.name = name
        self.positions=[]
        self.state=[]

    def __str__(self):
        return f"Chip(name={self.name}, description={self.description}, price={self.price}, stock={self.stock})"

    def get_positions(self):
        return self.positions