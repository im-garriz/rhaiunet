from trainers.trainer import Trainer
import os
from sys import platform

# pip list --format=freeze > requirements.txt


def main():

    trainer = Trainer("hyperparameters.yaml")
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    try:
        main()
        if platform == "linux" or platform == "linux2":
            os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        if platform == "linux" or platform == "linux2":
            os.system('chmod -R 777 .')
        exit()
