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
        
# Create empty text file
def create_txt(name):
    with open(name + "_setup" + '.txt', 'w') as f:
        f.write('')
        
# Write text file
def write_to_txt_file(file, information):
    with open(file, 'a') as f:
        f.write(str(information) + '\n')
    
        
def write_setup_to_txt_file(args, cuda, optimizer, scheduler, model, device):
        with open(args.save_csv + '_setup' + '.txt', 'a') as setup:
            setup.write("Setup: " +  str(args.save_csv) + '\n')
            setup.write('batch size' + ',' + str(args.batch_size) + '\n')
            setup.write('test batch size' + ',' + str(args.test_batch_size) + '\n')
            setup.write('epochs' + ',' + str(args.epochs) + '\n')
            setup.write('learning rate' + ',' + str(args.lr) + '\n')
            setup.write('gamma' + ',' + str(args.gamma) + '\n')
            setup.write('no cuda' + ',' + str(args.no_cuda) + '\n')
            setup.write('no mps' + ',' + str(args.no_mps) + '\n')
            setup.write('dry run' + ',' + str(args.dry_run) + '\n')
            setup.write('seed' + ',' + str(args.seed) + '\n')
            setup.write('log interval' + ',' + str(args.log_interval) + '\n')
            setup.write('cuda' + ',' + str(cuda) + '\n')
            setup.write('optimizer' + ',' + str(optimizer) + '\n')
            setup.write('scheduler' + ',' + str(scheduler) + '\n')
            setup.write('model' + ',' + str(model) + '\n')
            setup.write('device' + ',' + str(device) + '\n')
            
# Calculate duration time in minutes, seconds, milliseconds
def calculate_duration(start_time, end_time):
    duration = end_time - start_time
    duration_in_s = duration.total_seconds()
    seconds = duration_in_s
    minutes = duration_in_s / 60
    return minutes, seconds

# Print duration time in minutes, seconds, milliseconds
def training_duration(start_time, end_time):
    minutes, seconds = calculate_duration(start_time, end_time)
    return "Duration: " + str(minutes) + " minutes = " + str(seconds) + " seconds"
    