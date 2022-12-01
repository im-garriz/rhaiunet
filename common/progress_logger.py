import time

class ProgressBar:

    def __init__(self, steps):

        self.step = 0
        self.t0 = time.time()
        self.steps = steps
        # Longitud de la barra de progreso en caracteres
        self.bar_length = 40

    def step_bar(self):

        t1 = time.time()

        self.step += 1

        bar = self.get_updated_bar()

        if self.step > self.steps:
            print()
        elif self.step == self.steps:
            print("{} Step:{}/{} Elapsed time: {:.2f}s".format(bar, self.step, self.steps, t1 - self.t0),
                  end='\r')
            print()
        else:
            print("{} Step:{}/{} Elapsed time: {:.2f}s".format(bar, self.step, self.steps, t1 - self.t0),
                  end='\r')

    def get_updated_bar(self):
        """
        Generador y getter de la barra actual

        :return: string con la barra actual actualizada
        """
        percentage = self.step / self.steps
        progress_n = int(self.bar_length * percentage)

        bar = "|"
        for i in range(self.bar_length):
            if i <= progress_n:
                bar += "|"
            else:
                bar += " "
        bar += "|"

        return bar


class ProgressLogger:
    """
    Clase para imprimir log dinamicos con barras de progreso de cada epoch en pantalla
    """

    def __init__(self, train_n_batches, val_n_batches):
        """
        Constructor: inicializa todos los parametros a emplear
        :param train_n_batches: cantidad de batche de train
        :param val_n_batches: cantidad de batches de test
        """
        self.train_n_batches = train_n_batches
        self.val_n_batches = val_n_batches
        self.step = 0
        self.t0 = 0
        self.current_n_len = 0
        # Longitud de la barra de progreso en caracteres
        self.bar_length = 40

    def update_bar(self):
        """
        Metodo que actualiza la barra actual
        :return:
        """
        t1 = time.time()
        self.step += 1
        bar = self.get_updated_bar()

        if self.step > self.current_n_len:
            print()
        elif self.step == self.current_n_len:
            print("{} Step:{}/{} Elapsed time: {:.2f}s".format(bar, self.step, self.current_n_len, t1 - self.t0),
                  end='\r')
            print()
        else:
            print("{} Step:{}/{} Elapsed time: {:.2f}s".format(bar, self.step, self.current_n_len, t1 - self.t0),
                  end='\r')

    @staticmethod
    def print_epoch(epoch):
        """
        Metodo que imprime la epoch actual

        :param epoch: epoch actual
        :return:
        """
        print(f"Epoch: {epoch}")

    def get_updated_bar(self):
        """
        Generador y getter de la barra actual

        :return: string con la barra actual actualizada
        """
        percentage = self.step / self.current_n_len
        progress_n = int(self.bar_length * percentage)

        bar = "<"
        for i in range(self.bar_length):
            if i <= progress_n:
                bar += "="
            else:
                bar += " "
        bar += ">"

        return bar

    def initialize_bar(self, train):
        """
        Inicializador de la barra de progreso

        :param train: booleano que indica si es la de train o val
        :return:
        """
        self.t0 = time.time()
        self.step = -1

        if train:
            self.current_n_len = self.train_n_batches
        else:
            self.current_n_len = self.val_n_batches

        self.update_bar()

    def initialize_train_bar(self):
        """
        Inicializador de la barra de train

        :return:
        """
        print("Train step...")
        self.initialize_bar(train=True)

    def initialize_val_bar(self):
        """
        Inicializador de la barra de val

        :return:
        """

        print("Validation step...")
        self.initialize_bar(train=False)

    @staticmethod
    def finish_epoch():
        print()