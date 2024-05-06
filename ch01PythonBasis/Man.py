class Man:
    def __init__(self,name) -> None:
        self.name = name
        print("Initiallized!")
    def hello(self):
        print("Hello " + self.name + "!")
    def goodbye(self):
        print("goodbye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

