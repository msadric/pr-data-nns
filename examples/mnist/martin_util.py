# Utility class for this project

import datetime

# Create identifier (string) based on current date and time
def create_identifier():
    now = datetime.datetime.now()
    identifier = now.strftime("%d%m_%H%M%S")
    return identifier


# Create csv file with name (argument) with header train/test, epoch, loss, accuracy
def create_csv(name):
    with open(name + '.csv', 'w') as f:
        f.write('train/test, epoch,loss/avg_loss,accuracy\n')